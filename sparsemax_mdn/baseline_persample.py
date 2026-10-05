"""Per-sample sharpness evaluation of a BASELINE (softmax, base_mdn) checkpoint with the sparsemax harness.

    python -m sparsemax_mdn.baseline_persample --baseline-run RUN_DIR --label K3 --split validation [--limit N] --gpu 0
    python -m sparsemax_mdn.baseline_persample --baseline-run RUN_DIR --label K3 --split test --confirm-test-baseline

Same harness as sparsemax_mdn.evaluate (unchanged MDN_Forecaster.evaluate, captured per-sample arrays, same
summary). The baseline run directory is read-only; everything is written under
results/trained_models/sparsemax_mdn/baseline_persample/<label>/. One test evaluation per label.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402
import eval as base_eval  # noqa: E402
from eval import MDN_Forecaster  # noqa: E402
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import set_global_seed  # noqa: E402
from utils.mdn_distribution import apply_checkpoint_parameterization, build_mdn_distribution  # noqa: E402
from sparsemax_mdn import base_hashes  # noqa: E402
from sparsemax_mdn.evaluate import (OFFICIAL_KEYS, SEED, SPLITS, TEST_CLAIM, _FixedData,  # noqa: E402
                                    capture_plot_inputs, sharpness_per_sample_summary)

OUT_ROOT = ROOT / 'results/trained_models/sparsemax_mdn/baseline_persample'
BASE_CONFIG = ROOT / 'base_mdn/configs/imptc/default_peds_imptc.json'
ABLATION = ROOT / 'results/ablations/imptc_num_gaussians/ablation.json'
REPRO_FIELDS = ('ravg_percent', 'rmin_percent', 's68_m2_per_s', 's95_m2_per_s', 'asaee_m_per_s',
                'minade20_m', 'minfde20_m', 'test_nll')


# ----------------------------------------------------------------------------- pure helpers
def output_stem(split, limited):
    return f"{split}_best{'_limited' if limited else ''}"


def check_baseline_request(split, confirm_test_baseline, limit, out_dir=None):
    """Guard run BEFORE any data or checkpoint is read. out_dir=None checks the arguments only."""
    if split not in SPLITS:
        raise ValueError(f'split must be exactly one of {SPLITS}, got {split!r}')
    if limit is not None and limit <= 0:
        raise ValueError('--limit must be positive')
    if split == 'validation':
        if confirm_test_baseline:
            raise ValueError('--confirm-test-baseline only applies to --split test')
        return
    if limit is not None:
        raise ValueError('--limit is not allowed on the test split (one full test evaluation per label)')
    if not confirm_test_baseline:
        raise ValueError('--split test requires --confirm-test-baseline (one test evaluation per label)')
    if out_dir is not None:
        out_dir = Path(out_dir)
        existing = [p for p in (out_dir / f'{output_stem("test", False)}_persample.npz', out_dir / TEST_CLAIM)
                    if p.exists()]
        if existing:
            raise ValueError(f'A test evaluation already exists for this label, refusing: {[str(e) for e in existing]}')


def claim_test(out_dir, info):
    """Create the lock atomically (O_EXCL); raises FileExistsError if it already exists."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(out_dir / TEST_CLAIM, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as f:
        json.dump(info, f)


def select_ablation_entry(ablation, num_gaussians):
    matches = [r for r in ablation['results'] if r.get('num_gaussians') == num_gaussians]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one ablation entry with num_gaussians={num_gaussians}, found {len(matches)}')
    return matches[0]


def reproduction_diff(current, reference, fields=REPRO_FIELDS):
    """Per-field {current, reference, absolute_diff, relative_diff}; a missing value gives None entries."""
    out = {}
    for key in fields:
        cur, ref = current.get(key), reference.get(key)
        if cur is None or ref is None:
            out[key] = {'current': cur, 'reference': ref, 'absolute_diff': None, 'relative_diff': None}
            continue
        diff = float(cur) - float(ref)
        out[key] = {'current': float(cur), 'reference': float(ref), 'absolute_diff': diff,
                    'relative_diff': diff / abs(float(ref)) if ref != 0 else None}
    return out


@torch.no_grad()
def baseline_nll(model, X, y, num_gaussians, parameterization, input_horizon, batch_size, device):
    """NLL exactly as base_mdn/evaluate_run.py: batch mean of log_prob weighted by batch size, over n samples."""
    total = 0.0
    for start in range(0, len(X), batch_size):
        stop = min(len(X), start + batch_size)
        inputs = torch.as_tensor(X[start:stop, -input_horizon:], dtype=torch.float32, device=device)
        target = torch.as_tensor(y[start:stop], dtype=torch.float32, device=device)
        dist = build_mdn_distribution(model(inputs), num_gaussians, parameterization)
        total += float(-dist.log_prob(target).mean()) * (stop - start)
    return total / len(X)


# ----------------------------------------------------------------------------- CLI
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline-run', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--split', required=True, choices=SPLITS)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--confirm-test-baseline', action='store_true')
    parser.add_argument('--gpu', default='0')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.label or '/' in args.label or args.label.startswith('.'):
        raise ValueError(f'Invalid label {args.label!r}')
    check_baseline_request(args.split, args.confirm_test_baseline, args.limit)  # arguments only
    out_dir = OUT_ROOT / args.label
    check_baseline_request(args.split, args.confirm_test_baseline, args.limit, out_dir)
    run = Path(args.baseline_run).resolve()
    base_hashes.verify()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.split == 'test':  # claim before the first test byte is read
        claim_test(out_dir, {'baseline_run': str(run), 'label': args.label})

    kind = 'testing' if args.split == 'test' else 'eval'
    cfg = ConfigLoader(str(BASE_CONFIG), 'imptc', False, False, BASE_CONFIG.stem, 'base_mdn', kind)
    checkpoint_path = run / 'checkpoints/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint.get('resolved_config', {}).get('model_params', {})
    apply_checkpoint_parameterization(cfg, checkpoint)
    params = dict(saved) if saved else dict(cfg.model_params)
    params.setdefault('mdn_parameterization', cfg.model_params.get('mdn_parameterization'))
    model = LSTM_Trajectory_Forecast(params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    k = int(params['num_gaussians'])

    loader = DataLoader(cfg)
    if args.split == 'test':
        loader.load_test_data()
        arrays = tuple(loader.get_test_data())
    else:
        loader.load_eval_data()
        arrays = tuple(loader.eval_data[:5])
        if args.limit is not None:
            arrays = tuple(a[:args.limit] for a in arrays)
    X, y = arrays[0], arrays[1]
    nll = baseline_nll(model, X, y, k, params.get('mdn_parameterization'), cfg.test_params['num_input_horizons'],
                       cfg.test_params['batch_size'], device)

    scratch = out_dir / f'official_evaluator_{args.split}'
    for path in (scratch / 'examples/ego', scratch / 'examples/world'):
        path.mkdir(parents=True, exist_ok=True)
    cfg.evaluation_path = cfg.testing_path = str(scratch)
    cfg.eval_ego_examples_path = cfg.test_ego_examples_path = str(scratch / 'examples/ego')
    cfg.eval_world_examples_path = cfg.test_world_examples_path = str(scratch / 'examples/world')
    set_global_seed(SEED)
    forecaster = MDN_Forecaster(cfg, model, _FixedData(arrays), kind, logging.getLogger('baseline_persample'), device)
    captured = {}
    with capture_plot_inputs(base_eval, captured):
        metrics = forecaster.evaluate(epoch=checkpoint['epoch'])

    limited = args.limit is not None
    official = {key: metrics[key] for key in OFFICIAL_KEYS}
    summary = sharpness_per_sample_summary(captured['sharpness_area_m2'], cfg.test_params['test_horizons'],
                                           forecaster.dt, forecaster.forecast_horizon,
                                           cfg.test_params['confidence_levels'])
    result = {
        'baseline_run': str(run), 'label': args.label, 'checkpoint': str(checkpoint_path),
        'checkpoint_epoch': int(checkpoint['epoch']), 'num_gaussians': k, 'split': args.split,
        'num_samples': int(len(X)), 'flags': {'limit': args.limit, 'confirm_test_baseline': args.confirm_test_baseline},
        'seed': SEED, 'nll_evaluate_run_style': nll,
        'nll_note': 'softmax baseline: base build_mdn_distribution (Categorical(probs=pi)); exact and clamped coincide '
                    'because softmax weights are strictly positive',
        'official_metrics': official,
        'official_metrics_extra': {key: v for key, v in metrics.items() if key not in OFFICIAL_KEYS},
        'sharpness_per_sample_summary': summary,
        'official_percentile_grid_score': {name: s['official_percentile_grid_score'] for name, s in summary.items()},
        'evaluator_version': 'sparsemax_mdn.baseline_persample v1 (base MDN_Forecaster.evaluate unchanged)'}
    if args.split == 'test':
        reference = select_ablation_entry(json.loads(ABLATION.read_text()), k)
        result['reproduction_vs_ablation_json'] = reproduction_diff({**official, 'test_nll': nll}, reference)
        result['reproduction_reference'] = str(ABLATION)
    stem = output_stem(args.split, limited)
    np.savez_compressed(
        out_dir / f'{stem}_persample.npz', sample_index=np.arange(len(X)),
        confidence_levels=np.asarray(cfg.test_params['confidence_levels']),
        horizons=np.asarray(cfg.test_params['test_horizons']), delta_t=float(forecaster.dt),
        forecast_horizon=int(forecaster.forecast_horizon), **captured)
    out = out_dir / f'{stem}.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    base_hashes.verify()
    print(json.dumps(result, indent=2))
    print(f'wrote {out}')
    return result


if __name__ == '__main__':
    main()
