"""Evaluate the best checkpoint of a shared_decoder_mdn run with the unchanged baseline evaluator.

    python -m shared_decoder_mdn.evaluate --config CONFIG --run-id RUN --split validation --gpu 0
    python -m shared_decoder_mdn.evaluate --config CONFIG --run-id RUN --split test --confirm-test-once --gpu 0
    python -m shared_decoder_mdn.evaluate --baseline-run BASE_RUN_DIR --label K3 --split test --confirm-test-once --gpu 0

With --baseline-run the SAME code path evaluates a base_mdn checkpoint (LSTM_Trajectory_Forecast, default config);
output goes to results/trained_models/shared_decoder_mdn/baseline_eval/<label>/ and the baseline run is not written to.

Same procedure as base_mdn/evaluate_run.py (batch-size weighted NLL, MDN_Forecaster.evaluate, seed 2024). The test
split can be evaluated ONCE per run: a lock file is created atomically before the first test byte is read.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from eval import MDN_Forecaster  # noqa: E402
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import set_global_seed  # noqa: E402
from utils.mdn_distribution import apply_checkpoint_parameterization, build_mdn_distribution  # noqa: E402
from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402
from shared_decoder_mdn import base_hashes  # noqa: E402
from shared_decoder_mdn.model import SharedDecoderMDN  # noqa: E402

SEED = 2024
CONFIG_DIR = ROOT / 'shared_decoder_mdn/configs/imptc'
BASE_CONFIG = ROOT / 'base_mdn/configs/imptc/default_peds_imptc.json'
BASELINE_OUT = ROOT / 'results/trained_models/shared_decoder_mdn/baseline_eval'
ABLATION = ROOT / 'results/ablations/imptc_num_gaussians/ablation.json'
LOCK = 'test_evaluation_started.lock'
OFFICIAL_KEYS = ('ravg_percent', 'rmin_percent', 's68_m2_per_s', 's95_m2_per_s', 'asaee_m_per_s',
                 'minade20_m', 'minfde20_m')


class _FixedData:
    """Hands the forecaster exactly the arrays chosen here."""

    def __init__(self, arrays):
        self.arrays = arrays

    def get_eval_data(self):
        return self.arrays

    def get_test_data(self):
        return self.arrays


def check_request(split, confirm_test_once, eval_dir=None):
    if split not in ('validation', 'test'):
        raise ValueError(f'split must be validation or test, got {split!r}')
    if split == 'validation':
        if confirm_test_once:
            raise ValueError('--confirm-test-once only applies to --split test')
        return
    if not confirm_test_once:
        raise ValueError('--split test requires --confirm-test-once (one test evaluation per run)')
    if eval_dir is not None and ((Path(eval_dir) / LOCK).exists() or (Path(eval_dir) / 'test_best.json').exists()):
        raise ValueError('A test evaluation already exists for this run, refusing')


@torch.no_grad()
def nll_of(model, X, y, num_gaussians, parameterization, input_horizon, batch_size, device):
    total = 0.0
    for start in range(0, len(X), batch_size):
        stop = min(len(X), start + batch_size)
        inputs = torch.as_tensor(X[start:stop, -input_horizon:], dtype=torch.float32, device=device)
        target = torch.as_tensor(y[start:stop], dtype=torch.float32, device=device)
        dist = build_mdn_distribution(model(inputs), num_gaussians, parameterization)
        total += float(-dist.log_prob(target).mean()) * (stop - start)
    return total / len(X)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config')
    parser.add_argument('--run-id')
    parser.add_argument('--baseline-run')
    parser.add_argument('--label')
    parser.add_argument('--split', required=True, choices=('validation', 'test'))
    parser.add_argument('--confirm-test-once', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args(argv)
    baseline = args.baseline_run is not None
    if baseline:
        if not args.label or '/' in args.label or args.label.startswith('.') or args.config or args.run_id:
            raise ValueError('--baseline-run needs --label and no --config/--run-id')
    elif not args.config or not args.run_id:
        raise ValueError('--config and --run-id are required (or use --baseline-run with --label)')
    check_request(args.split, args.confirm_test_once)
    kind = 'testing' if args.split == 'test' else 'eval'
    if baseline:
        name = BASE_CONFIG.name
        cfg = ConfigLoader(str(BASE_CONFIG), 'imptc', False, False, BASE_CONFIG.stem, 'base_mdn', kind)
        run = Path(args.baseline_run).resolve()
        eval_dir = BASELINE_OUT / args.label
    else:
        name = args.config if args.config.endswith('.json') else args.config + '.json'
        cfg = ConfigLoader(str(CONFIG_DIR / name), 'imptc', False, False, name[:-5], 'shared_decoder_mdn', kind)
        run = Path(cfg.result_path) / 'runs' / args.run_id
        eval_dir = run / 'evaluation'
    check_request(args.split, args.confirm_test_once, eval_dir)
    base_hashes.verify()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
    eval_dir.mkdir(parents=True, exist_ok=True)
    if args.split == 'test':  # claim before the first test byte is read
        fd = os.open(eval_dir / LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            json.dump({'run': str(run), 'config': name}, f)

    checkpoint_path = run / 'checkpoints/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint.get('resolved_config', {}).get('model_params')
    if baseline:
        apply_checkpoint_parameterization(cfg, checkpoint)
        params = dict(saved) if saved else dict(cfg.model_params)
        params.setdefault('mdn_parameterization', cfg.model_params.get('mdn_parameterization'))
        cfg.model_params['num_gaussians'] = int(params['num_gaussians'])
        model = LSTM_Trajectory_Forecast(params).to(device)
    else:
        if saved is not None and saved != cfg.model_params:
            raise ValueError('Checkpoint model_params differ from config')
        model = SharedDecoderMDN(cfg.model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    loader = DataLoader(cfg)
    if args.split == 'test':
        loader.load_test_data()
        arrays = tuple(loader.get_test_data())
    else:
        loader.load_eval_data()
        arrays = tuple(loader.eval_data[:5])
    X, y = arrays[0], arrays[1]
    k = cfg.model_params['num_gaussians']
    nll = nll_of(model, X, y, k, cfg.model_params.get('mdn_parameterization'), cfg.test_params['num_input_horizons'],
                 cfg.test_params['batch_size'], device)

    scratch = eval_dir / f'official_evaluator_{args.split}'
    for path in (scratch / 'examples/ego', scratch / 'examples/world'):
        path.mkdir(parents=True, exist_ok=True)
    cfg.evaluation_path = cfg.testing_path = str(scratch)
    cfg.eval_ego_examples_path = cfg.test_ego_examples_path = str(scratch / 'examples/ego')
    cfg.eval_world_examples_path = cfg.test_world_examples_path = str(scratch / 'examples/world')
    set_global_seed(SEED)
    forecaster = MDN_Forecaster(cfg, model, _FixedData(arrays), kind, logging.getLogger('shared_decoder_eval'), device)
    metrics = forecaster.evaluate(epoch=checkpoint['epoch'])
    result = {'run_id': args.run_id or args.label, 'run_dir': str(run), 'config': name, 'baseline': baseline, 'split': args.split, 'checkpoint': str(checkpoint_path),
              'checkpoint_epoch': int(checkpoint['epoch']), 'num_samples': int(len(X)), 'seed': SEED,
              'parameter_count': sum(p.numel() for p in model.parameters()), f'{args.split}_nll': nll,
              'official_metrics': {key: metrics[key] for key in OFFICIAL_KEYS},
              'official_metrics_extra': {key: v for key, v in metrics.items() if key not in OFFICIAL_KEYS}}
    if baseline and args.split == 'test':
        ref = [r for r in json.loads(ABLATION.read_text())['results'] if r.get('num_gaussians') == k]
        if len(ref) == 1:
            ref = ref[0]
            current = {**result['official_metrics'], 'test_nll': nll}
            result['reproduction_vs_ablation_json'] = {
                key: {'current': current[key], 'reference': ref.get(key),
                      'absolute_diff': None if ref.get(key) is None else current[key] - ref[key]}
                for key in (*OFFICIAL_KEYS, 'test_nll')}
    out = eval_dir / f'{args.split}_best.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    base_hashes.verify()
    print(json.dumps({k_: v for k_, v in result.items() if k_ != 'official_metrics_extra'}, indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
