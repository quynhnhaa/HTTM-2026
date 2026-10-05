"""Evaluate the best checkpoint of a crps_mdn run with the unchanged baseline evaluator.

    python -m crps_mdn.evaluate --config crps_peds_imptc --run-id crps_seed2024 --split validation --gpu 0
    python -m crps_mdn.evaluate --config crps_peds_imptc --run-id crps_seed2024 --split test --confirm-test-once --gpu 0

Same procedure as base_mdn/evaluate_run.py (batch-size weighted NLL, MDN_Forecaster.evaluate, seed 2024). The model is the
baseline architecture (the CRPS term only changes training). The test split can be evaluated ONCE per run: a lock file is
created atomically before the first test byte is read.
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
from crps_mdn import base_hashes  # noqa: E402

SEED = 2024
CONFIG_DIR = ROOT / 'crps_mdn/configs/imptc'
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
    parser.add_argument('--split', required=True, choices=('validation', 'test'))
    parser.add_argument('--confirm-test-once', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args(argv)
    if not args.config or not args.run_id:
        raise ValueError('--config and --run-id are required')
    check_request(args.split, args.confirm_test_once)
    kind = 'testing' if args.split == 'test' else 'eval'
    name = args.config if args.config.endswith('.json') else args.config + '.json'
    cfg = ConfigLoader(str(CONFIG_DIR / name), 'imptc', False, False, name[:-5], 'crps_mdn', kind)
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
    if saved is not None and saved != cfg.model_params:
        raise ValueError('Checkpoint model_params differ from config')
    apply_checkpoint_parameterization(cfg, checkpoint)
    model = LSTM_Trajectory_Forecast(cfg.model_params).to(device)
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
    result = {'run_id': args.run_id, 'run_dir': str(run), 'config': name, 'split': args.split, 'checkpoint': str(checkpoint_path),
              'checkpoint_epoch': int(checkpoint['epoch']), 'num_samples': int(len(X)), 'seed': SEED,
              'parameter_count': sum(p.numel() for p in model.parameters()), f'{args.split}_nll': nll,
              'official_metrics': {key: metrics[key] for key in OFFICIAL_KEYS},
              'official_metrics_extra': {key: v for key, v in metrics.items() if key not in OFFICIAL_KEYS}}
    out = eval_dir / f'{args.split}_best.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    base_hashes.verify()
    print(json.dumps({k_: v for k_, v in result.items() if k_ != 'official_metrics_extra'}, indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
