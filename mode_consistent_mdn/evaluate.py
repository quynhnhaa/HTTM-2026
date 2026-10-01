"""Evaluate variant marginals and trajectory-coherent samples on IMPTC test."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'base_mdn'))

from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402
from eval import MDN_Forecaster  # noqa: E402
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import set_global_seed  # noqa: E402
from utils.mdn_distribution import build_mdn_distribution, decode_mdn_output  # noqa: E402

from model import ModeConsistentMDN, joint_nll_per_step, sample_coherent_trajectories  # noqa: E402


def checkpoint_model(path, model, device, expected_family):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    resolved = checkpoint.get('resolved_config', {})
    loss_name = resolved.get('experiment_params', {}).get('loss_name')
    if expected_family == 'variant' and loss_name != 'joint_trajectory_nll_divided_by_horizon':
        raise ValueError(f'Not a mode-consistent variant checkpoint: {path}')
    if expected_family == 'baseline' and loss_name == 'joint_trajectory_nll_divided_by_horizon':
        raise ValueError(f'Not a baseline checkpoint: {path}')
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint['model']))
    return model.to(device).eval(), int(checkpoint['epoch'])


@torch.no_grad()
def supplemental_metrics(model, data_X, data_y, cfg, device, coherent, limit=None):
    """Same K, seed, horizons, and distance formula; sample semantics differ."""
    batch_size = int(cfg.test_params['batch_size'])
    horizon_indices = cfg.test_params['test_horizons']
    k = int(cfg.test_params['num_k_samples'])
    num_gaussians = int(cfg.model_params['num_gaussians'])
    n = len(data_X) if limit is None else min(int(limit), len(data_X))
    if n <= 0:
        raise ValueError('No test samples selected')
    sum_min_ade = sum_min_fde = sum_marginal_nll = 0.0
    sum_joint_nll_per_step = 0.0
    sum_pi = torch.zeros(num_gaussians, device=device)
    argmax_counts = torch.zeros(num_gaussians, device=device)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        X = torch.as_tensor(
            data_X[start:stop, -cfg.test_params['num_input_horizons']:],
            dtype=torch.float32, device=device,
        )
        y = torch.as_tensor(data_y[start:stop], dtype=torch.float32, device=device)
        output = model(X)
        distribution = build_mdn_distribution(output, num_gaussians, cfg.model_params.get('mdn_parameterization'))
        sum_marginal_nll += float(-distribution.log_prob(y).mean()) * (stop - start)
        if coherent:
            samples, _ = sample_coherent_trajectories(output, num_gaussians, k, cfg.model_params.get('mdn_parameterization'))
            sum_joint_nll_per_step += float(
                joint_nll_per_step(output, y, num_gaussians, cfg.model_params.get('mdn_parameterization'))
            ) * (stop - start)
            pi = decode_mdn_output(output, num_gaussians, cfg.model_params.get('mdn_parameterization'))['pi'][:, 0, :]
            sum_pi += pi.sum(dim=0)
            argmax_counts += torch.bincount(pi.argmax(dim=-1), minlength=num_gaussians)
        else:
            samples = distribution.sample((k,))

        errors = torch.linalg.vector_norm(
            samples[:, :, horizon_indices, :] - y[None, :, horizon_indices, :],
            dim=-1,
        )
        sum_min_ade += float(errors.mean(dim=-1).min(dim=0).values.sum())
        sum_min_fde += float(errors[:, :, -1].min(dim=0).values.sum())

    result = {
        'sample_count': n,
        'num_trajectory_samples': k,
        'test_horizons': horizon_indices,
        'sample_semantics': 'one_mode_per_trajectory' if coherent else 'independent_marginal_at_each_horizon',
        'minade20_m': sum_min_ade / n,
        'minfde20_m': sum_min_fde / n,
        'marginal_nll_per_position': sum_marginal_nll / n,
    }
    if coherent:
        result['joint_nll_per_step'] = sum_joint_nll_per_step / n
        result['mean_mode_weights'] = (sum_pi / n).tolist()
        result['argmax_mode_fraction'] = (argmax_counts / n).tolist()
    return result


def main(args):
    if args.official and args.limit:
        raise ValueError('--official requires the complete test split; omit --limit')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    set_global_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / 'mode_consistent_mdn/configs/imptc' / config_path
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = ConfigLoader(
        str(config_path), 'imptc', False, False,
        config_path.stem, 'mode_consistent_mdn', 'testing',
    )
    run_dir = Path(cfg.result_path) / 'runs' / args.run_id
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / 'checkpoints/best.pt'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model, epoch = checkpoint_model(
        checkpoint_path, ModeConsistentMDN(cfg.model_params), device, 'variant'
    )

    from utils.mdn_distribution import apply_checkpoint_parameterization
    apply_checkpoint_parameterization(cfg, torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    data_loader = DataLoader(cfg)
    data_loader.load_test_data()
    data_X, data_y, _, _, _ = data_loader.get_test_data()
    output_dir = run_dir / 'testing'
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    variant = supplemental_metrics(model, data_X, data_y, cfg, device, True, args.limit)
    report = {
        'schema_version': '1.0',
        'split': 'test',
        'run_id': args.run_id,
        'checkpoint': str(checkpoint_path.resolve()),
        'epoch': epoch,
        'evaluation_seed': args.seed,
        'variant_parameter_count': sum(p.numel() for p in model.parameters()),
        'variant_trajectory_sampling': variant,
    }

    if args.baseline_checkpoint:
        baseline_path = Path(args.baseline_checkpoint)
        baseline_model, baseline_epoch = checkpoint_model(
            baseline_path,
            LSTM_Trajectory_Forecast(cfg.model_params),
            device, 'baseline',
        )
        from utils.mdn_distribution import checkpoint_parameterization, resolve_parameterization
        baseline_policy = checkpoint_parameterization(torch.load(baseline_path, map_location='cpu', weights_only=False))
        if baseline_policy != resolve_parameterization(cfg.model_params.get('mdn_parameterization')):
            raise ValueError('Baseline and variant use different MDN parameterizations')
        set_global_seed(args.seed)
        report['baseline_independent_sampling'] = supplemental_metrics(
            baseline_model, data_X, data_y, cfg, device, False, args.limit
        )
        report['baseline_checkpoint'] = str(baseline_path.resolve())
        report['baseline_epoch'] = baseline_epoch
        report['baseline_parameter_count'] = sum(p.numel() for p in baseline_model.parameters())

    if args.official:
        # The original evaluator reads one 2D GMM per horizon. Its ADE/FDE
        # sampling convention is left intact and is named separately below.
        official_dir = output_dir / 'official_marginal_evaluator'
        official_dir.mkdir(parents=True, exist_ok=True)
        cfg.testing_path = str(official_dir)
        cfg.test_ego_examples_path = str(official_dir / 'examples/ego')
        cfg.test_world_examples_path = str(official_dir / 'examples/world')
        Path(cfg.test_ego_examples_path).mkdir(parents=True, exist_ok=True)
        Path(cfg.test_world_examples_path).mkdir(parents=True, exist_ok=True)
        set_global_seed(args.seed)
        forecaster = MDN_Forecaster(
            cfg=cfg, model=model, data_loader=data_loader,
            type='testing', device=device, logger=logging.getLogger(__name__),
        )
        report['repository_official_marginal_metrics'] = forecaster.evaluate(epoch=epoch)

    report_path = output_dir / ('evaluation_limited.json' if args.limit else 'evaluation.json')
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'Saved evaluation: {report_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='mode_consistent_peds_imptc.json',
                        help='Variant config filename or absolute path')
    parser.add_argument('--checkpoint', help='Variant checkpoint; defaults to run best.pt')
    parser.add_argument('--baseline-checkpoint', help='Optional read-only baseline M3 checkpoint')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--limit', type=int, help='Labelled quick analysis of first N test samples')
    parser.add_argument('--official', action='store_true', help='Run full repository marginal evaluator')
    main(parser.parse_args())
