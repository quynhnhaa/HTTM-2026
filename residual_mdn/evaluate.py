"""Evaluate residual MDN with the repository's unchanged marginal GMM evaluator."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stable_mdn'))

from eval import MDN_Forecaster  # noqa: E402
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import set_global_seed  # noqa: E402
from utils.mdn_distribution import build_mdn_distribution  # noqa: E402
from model import ResidualMDN  # noqa: E402


@torch.no_grad()
def marginal_nll(model, X, y, cfg, device, limit=None):
    n = len(X) if limit is None else min(len(X), limit)
    if n < 1:
        raise ValueError('Empty test selection')
    total = 0.0
    for start in range(0, n, int(cfg.test_params['batch_size'])):
        stop = min(start + int(cfg.test_params['batch_size']), n)
        inputs = torch.as_tensor(X[start:stop, -cfg.test_params['num_input_horizons']:],
                                 dtype=torch.float32, device=device)
        target = torch.as_tensor(y[start:stop], dtype=torch.float32, device=device)
        output = model(inputs)
        total += float(-build_mdn_distribution(
            output, cfg.model_params['num_gaussians'], cfg.model_params.get('mdn_parameterization')
        ).log_prob(target).mean()) * (stop - start)
    return total / n


def main(args):
    if args.official and args.limit:
        raise ValueError('--official needs the full test split')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    set_global_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / 'residual_mdn/configs/imptc' / config_path
    cfg = ConfigLoader(str(config_path), 'imptc', False, False,
                       config_path.stem, 'residual_mdn', 'testing')
    run = Path(cfg.result_path) / 'runs' / args.run_id
    checkpoint_path = run / 'checkpoints/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from utils.mdn_distribution import apply_checkpoint_parameterization
    cfg.experiment_params['evaluation_seed'] = args.seed
    apply_checkpoint_parameterization(cfg, checkpoint)
    resolved = checkpoint.get('resolved_config', {})
    if resolved.get('experiment_params', {}).get('loss_name') != 'marginal_nll':
        raise ValueError('Checkpoint does not belong to residual MDN')
    if {k:v for k,v in resolved.get('model_params', {}).items() if k != 'mdn_parameterization'} != {k:v for k,v in cfg.model_params.items() if k != 'mdn_parameterization'}:
        raise ValueError('Checkpoint and config model parameters differ')
    model = ResidualMDN(cfg.model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    loader = DataLoader(cfg)
    loader.load_test_data()
    X, y, _, _, _ = loader.get_test_data()
    n = len(X) if args.limit is None else min(len(X), args.limit)
    report = {
        'evaluator_version': 'stable_v2_ecdf_rng',
        'schema_version': '1.0', 'split': 'test' if args.limit is None else 'test_limited',
        'sample_count': n, 'run_id': args.run_id, 'seed': args.seed,
        'checkpoint': str(checkpoint_path.resolve()), 'epoch': int(checkpoint['epoch']),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'mdn_parameterization': cfg.model_params['mdn_parameterization'],
        'test_params': cfg.test_params, 'eval_metrics': cfg.eval_metrics,
        'test_nll': marginal_nll(model, X, y, cfg, device, args.limit),
    }
    output_dir = run / 'testing'
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.official:
        official_dir = output_dir / 'official_evaluator'
        official_dir.mkdir(parents=True, exist_ok=True)
        cfg.testing_path = str(official_dir)
        cfg.test_ego_examples_path = str(official_dir / 'examples/ego')
        cfg.test_world_examples_path = str(official_dir / 'examples/world')
        Path(cfg.test_ego_examples_path).mkdir(parents=True, exist_ok=True)
        Path(cfg.test_world_examples_path).mkdir(parents=True, exist_ok=True)
        set_global_seed(args.seed)
        forecaster = MDN_Forecaster(
            cfg=cfg, model=model, data_loader=loader, type='testing',
            device=device, logger=logging.getLogger(__name__),
        )
        report['official_metrics'] = forecaster.evaluate(epoch=int(checkpoint['epoch']))
    if args.baseline_evaluation:
        if not args.official:
            raise ValueError('--baseline-evaluation requires --official')
        from utils.mdn_distribution import resolve_parameterization
        baseline_path = Path(args.baseline_evaluation)
        baseline = json.loads(baseline_path.read_text())
        if baseline.get('evaluator_version') != 'stable_v2_ecdf_rng':
            raise ValueError('Baseline evaluator version differs')
        if resolve_parameterization(baseline.get('mdn_parameterization')) != cfg.model_params['mdn_parameterization']:
            raise ValueError('Baseline and residual use different MDN parameterizations')
        if baseline.get('split') != 'test' or baseline.get('sample_count') != n or baseline.get('seed') != args.seed:
            raise ValueError('Baseline test split, sample count or evaluation seed differs')
        if baseline.get('test_params') != cfg.test_params or baseline.get('eval_metrics') != cfg.eval_metrics:
            raise ValueError('Baseline and residual evaluation protocols differ')
        report['baseline_source'] = str(baseline_path.resolve())
        report['baseline_m3'] = baseline
        report['delta_residual_minus_baseline'] = {
            key: float(value) - float(baseline['official_metrics'][key])
            for key, value in report['official_metrics'].items()
            if key in baseline['official_metrics'] and isinstance(value, (int, float))
        }
        report['delta_residual_minus_baseline']['test_nll'] = report['test_nll'] - baseline['test_nll']
    path = output_dir / ('evaluation_limited.json' if args.limit else 'evaluation.json')
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'Saved evaluation: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='residual_peds_imptc.json')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--baseline-evaluation', help='Baseline evaluation.json from the same parameterization')
    parser.add_argument('--official', action='store_true')
    main(parser.parse_args())
