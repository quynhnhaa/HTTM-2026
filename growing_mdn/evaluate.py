"""Evaluate the selected K on the test split with the legacy repository metrics.

Run only on explicit user request, once, after the selection rule is frozen.
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
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import set_global_seed  # noqa: E402
from eval import MDN_Forecaster  # noqa: E402
from growing_mdn.artifacts import ARCH  # noqa: E402
from growing_mdn.model import build_model  # noqa: E402
from growing_mdn.train import validation_nll  # noqa: E402


def check_official_request(k, selection, output_path, run_id=None):
    """Guard for the single official test run: only the selected K, only once per run."""
    if selection is None:
        raise ValueError('--official requires selection.json (run growing_mdn.select_k first)')
    if run_id is not None and selection.get('run_id') != run_id:
        raise ValueError(f"selection.json belongs to run {selection.get('run_id')}, not {run_id}")
    if k is None:
        k = selection['selected_k']
    if k != selection['selected_k']:
        raise ValueError(f"--official is only allowed for the selected K={selection['selected_k']}, got K={k}")
    output_path = Path(output_path)
    existing = sorted(output_path.parent.glob('evaluation_k??.json')) if output_path.parent.exists() else []
    if existing:
        raise ValueError(f'Official test result already exists for this run, refusing: {[str(e) for e in existing]}')
    return k


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='growing_peds_imptc.json')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--k', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--official', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    if args.official == (args.limit is not None):
        raise ValueError('Pass exactly one of --official (full test split) or --limit N')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_global_seed(2024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = ROOT / 'growing_mdn/configs/imptc' / args.config
    cfg = ConfigLoader(str(path), 'imptc', False, False, path.stem, 'growing_mdn', 'testing')
    run = Path(cfg.result_path) / 'runs' / args.run_id
    selection_path = run / 'selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    if args.official:
        k = check_official_request(args.k, selection, run / 'testing/evaluation_k00.json', args.run_id)
    else:
        k = args.k or (selection or {}).get('selected_k')
        if k is None:
            raise ValueError('--k is required when selection.json does not exist')
    checkpoint = torch.load(run / f'checkpoints/phase_k{k:02d}.pt', map_location=device, weights_only=False)
    if checkpoint.get('architecture') != ARCH or checkpoint['num_gaussians'] != k:
        raise ValueError(f'Checkpoint is not a K={k} growing_mdn phase checkpoint')
    cfg.model_params['num_gaussians'] = k
    model = build_model(cfg.model_params, k, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    loader = DataLoader(cfg)
    if args.official:
        loader.load_test_data()
        X, y = loader.get_test_data()[:2]
    else:  # pipeline check on VALIDATION data only; test data is touched only by --official
        loader.load_eval_data()
        X, y = loader.eval_data[0], loader.eval_data[1]
    n = len(X) if args.limit is None else min(args.limit, len(X))
    result = {'run_id': args.run_id, 'k': k, 'epoch': checkpoint['epoch'],
              'split': 'test' if args.official else 'validation_limited', 'sample_count': n, 'seed': 2024,
              'mdn_parameterization': {'mode': 'legacy'}, 'evaluator_version': 'base_mdn_legacy_bins',
              'test_params': cfg.test_params, 'parameter_count': sum(p.numel() for p in model.parameters()),
              ('test_nll' if args.official else 'validation_nll'):
                  validation_nll(model, X[:n], y[:n], device, cfg.test_params['batch_size'], k)}
    if args.official:
        cfg.testing_path = str(run / f'testing/official_k{k:02d}')
        Path(cfg.testing_path).mkdir(parents=True, exist_ok=True)
        set_global_seed(2024)
        forecaster = MDN_Forecaster(cfg, model, loader, 'testing', logging.getLogger('growing'), device)
        result['official_metrics'] = forecaster.evaluate(epoch=checkpoint['epoch'])
    destination = run / 'testing'
    destination.mkdir(exist_ok=True)
    out = destination / (f'evaluation_k{k:02d}.json' if args.official else f'evaluation_k{k:02d}_validation_limited.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
