"""Evaluate a baseline best checkpoint, preserving its distribution policy."""
import argparse
import json
import logging
import os
from pathlib import Path

import torch

from base_lstm import LSTM_Trajectory_Forecast
from eval import MDN_Forecaster
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.experiment import set_global_seed
from utils.mdn_distribution import apply_checkpoint_parameterization, build_mdn_distribution

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    if args.official and args.limit:
        raise ValueError('--official requires the full test split')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_global_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / 'stable_mdn/configs/imptc' / path
    cfg = ConfigLoader(str(path), 'imptc', False, False, path.stem, 'stable_mdn', 'testing')
    run = Path(cfg.result_path) / 'runs' / args.run_id
    checkpoint_path = run / 'checkpoints/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint.get('resolved_config', {}).get('model_params', {})
    architecture = lambda params: {k: v for k, v in params.items() if k != 'mdn_parameterization'}
    if saved and architecture(saved) != architecture(cfg.model_params):
        raise ValueError('Checkpoint architecture differs from config')
    cfg.experiment_params['evaluation_seed'] = args.seed
    apply_checkpoint_parameterization(cfg, checkpoint)
    model = LSTM_Trajectory_Forecast(cfg.model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    loader = DataLoader(cfg)
    loader.load_test_data()
    X, y, _, _, _ = loader.get_test_data()
    n = len(X) if args.limit is None else min(args.limit, len(X))
    if n <= 0:
        raise ValueError('Empty test selection')
    total = 0.0
    with torch.no_grad():
        for start in range(0, n, cfg.test_params['batch_size']):
            stop = min(n, start + cfg.test_params['batch_size'])
            inputs = torch.as_tensor(X[start:stop, -cfg.test_params['num_input_horizons']:], dtype=torch.float32, device=device)
            target = torch.as_tensor(y[start:stop], dtype=torch.float32, device=device)
            distribution = build_mdn_distribution(model(inputs), cfg.model_params['num_gaussians'], cfg.model_params['mdn_parameterization'])
            total += float(-distribution.log_prob(target).mean()) * (stop - start)
    report = {
        'evaluator_version': 'stable_v2_ecdf_rng',
        'schema_version': '1.0', 'split': 'test' if args.limit is None else 'test_limited',
        'sample_count': n, 'run_id': args.run_id, 'seed': args.seed,
        'checkpoint': str(checkpoint_path.resolve()), 'epoch': int(checkpoint['epoch']),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'mdn_parameterization': cfg.model_params['mdn_parameterization'],
        'test_params': cfg.test_params, 'eval_metrics': cfg.eval_metrics,
        'test_nll': total / n,
    }
    output = run / 'testing'
    output.mkdir(exist_ok=True)
    if args.official:
        directory = output / 'official_evaluator'
        directory.mkdir(exist_ok=True)
        cfg.testing_path = str(directory)
        cfg.test_ego_examples_path = str(directory / 'examples/ego')
        cfg.test_world_examples_path = str(directory / 'examples/world')
        Path(cfg.test_ego_examples_path).mkdir(parents=True, exist_ok=True)
        Path(cfg.test_world_examples_path).mkdir(parents=True, exist_ok=True)
        set_global_seed(args.seed)
        report['official_metrics'] = MDN_Forecaster(cfg, model, loader, 'testing', logging.getLogger(__name__), device).evaluate(epoch=checkpoint['epoch'])
    destination = output / ('evaluation_limited.json' if args.limit else 'evaluation.json')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved evaluation: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='default_peds_imptc.json')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--official', action='store_true')
    main(parser.parse_args())
