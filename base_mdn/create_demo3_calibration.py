#!/usr/bin/env python3
"""Create Demo 3 calibration artifacts from the trained IMPTC M=3 baseline."""

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from base_lstm import LSTM_Trajectory_Forecast
from eval import MDN_Forecaster
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.experiment import set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / (
    'results/trained_models/base_mdn/imptc/default_peds_imptc/'
    'runs/imptc_baseline_seed2024'
)
DEFAULT_CONFIG = PROJECT_ROOT / 'base_mdn/configs/imptc/default_peds_imptc.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the IMPTC M=3 reliability calibration plot from best.pt.'
    )
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--mc-samples', type=int, default=None)
    parser.add_argument(
        '--max-test-samples', type=int, default=None,
        help='Development smoke-test only. Omit for the slide/report artifact.',
    )
    return parser.parse_args()


def calibration_curves(confidence_sets, bins):
    """Match the repository's calibration binning and score definitions."""
    confidence_sets = np.asarray(confidence_sets)
    bins = np.asarray(bins)
    if confidence_sets.ndim != 2:
        raise ValueError('confidence_sets must have shape [samples, horizons]')
    bin_data = np.digitize(confidence_sets, bins=bins)
    observed = []
    errors = []
    for horizon in range(confidence_sets.shape[1]):
        counts = np.bincount(bin_data[:, horizon], minlength=len(bins) + 1)
        frequency = np.cumsum(counts[1:]) / confidence_sets.shape[0]
        observed.append(frequency)
        errors.append(np.abs(frequency - bins))
    observed = np.asarray(observed)
    errors = np.asarray(errors)
    return {
        'observed': observed,
        'errors': errors,
        'ravg_percent': float((1.0 - errors.mean()) * 100.0),
        'rmin_percent': float((1.0 - errors.max()) * 100.0),
        'ravg_by_horizon_percent': (1.0 - errors.mean(axis=1)) * 100.0,
        'rmin_by_horizon_percent': (1.0 - errors.max(axis=1)) * 100.0,
    }


def load_inputs(args, device):
    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = run_dir / 'checkpoints' / 'best.pt'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Missing validation-best checkpoint: {checkpoint_path}')
    cfg = ConfigLoader(
        config_path=str(config_path), target='imptc', with_log=False,
        with_print=True, name=config_path.stem, model_arch='base_mdn', type='testing',
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_params = checkpoint.get('resolved_config', {}).get('model_params')
    if saved_params != cfg.model_params:
        raise ValueError('Checkpoint model_params do not match the active IMPTC config')
    if cfg.model_params['num_gaussians'] != 3:
        raise ValueError(f"Demo 3 requires M=3, got M={cfg.model_params['num_gaussians']}")
    loader = DataLoader(cfg=cfg)
    loader.load_test_data()
    model = LSTM_Trajectory_Forecast(cfg.model_params).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return run_dir, checkpoint_path, checkpoint, cfg, loader, model


def collect_confidence_sets(cfg, loader, model, device, mc_samples, max_samples=None):
    data_x, data_y, _, _, _ = loader.get_test_data()
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError('--max-test-samples must be positive')
        data_x, data_y = data_x[:max_samples], data_y[:max_samples]
    horizons = cfg.test_params['test_horizons']
    batch_size = cfg.test_params['batch_size']
    helper = MDN_Forecaster(
        cfg=cfg, model=model, data_loader=loader, type='testing', device=device,
        logger=logging.getLogger('demo3_calibration'),
    )
    confidence_sets = []
    with torch.no_grad():
        for start in trange(0, len(data_x), batch_size, desc='Demo 3 reliability'):
            x = torch.as_tensor(
                data_x[start:start + batch_size, -cfg.test_params['num_input_horizons']:],
                dtype=torch.float32, device=device,
            )
            target = torch.as_tensor(
                data_y[start:start + batch_size], dtype=torch.float32, device=device,
            )
            output = model(x)
            output = torch.stack([output[:, step] for step in horizons], dim=1)
            target = torch.stack([target[:, step] for step in horizons], dim=1)
            confidence_sets.append(helper.build_confidence_set_mdn(
                output=output, target=target,
                num_gaussians=cfg.model_params['num_gaussians'],
                n_samples=mc_samples,
            ).cpu().numpy())
    return np.vstack(confidence_sets)


def save_csvs(output_dir, bins, times, curves):
    with (output_dir / 'calibration_curve.csv').open('w', newline='') as stream:
        fields = ['expected_confidence'] + [f'observed_{time:.1f}s' for time in times]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, expected in enumerate(bins):
            row = {'expected_confidence': float(expected)}
            row.update({
                f'observed_{time:.1f}s': float(curves['observed'][horizon, index])
                for horizon, time in enumerate(times)
            })
            writer.writerow(row)
    with (output_dir / 'reliability_by_horizon.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'horizon_seconds', 'ravg_percent', 'rmin_percent'
        ])
        writer.writeheader()
        for index, time in enumerate(times):
            writer.writerow({
                'horizon_seconds': time,
                'ravg_percent': float(curves['ravg_by_horizon_percent'][index]),
                'rmin_percent': float(curves['rmin_by_horizon_percent'][index]),
            })


def plot_calibration(output_dir, bins, times, curves, sample_count, mc_samples):
    colors = plt.get_cmap('tab10').colors
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.plot(bins, bins, color='black', linestyle='--', linewidth=2.2,
            label='Ideal calibration')
    for index, time in enumerate(times):
        ax.plot(
            bins, curves['observed'][index], linewidth=2.0,
            color=colors[index % len(colors)],
            label=(f'+{time:.1f} s  |  '
                   f'Ravg={curves["ravg_by_horizon_percent"][index]:.1f}%  '
                   f'Rmin={curves["rmin_by_horizon_percent"][index]:.1f}%'),
        )
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Expected confidence level')
    ax.set_ylabel('Observed frequency')
    ax.set_title(
        'Demo 3 — Reliability Calibration of IMPTC Baseline (M=3)\n'
        f'Overall Ravg={curves["ravg_percent"]:.1f}% · '
        f'Rmin={curves["rmin_percent"]:.1f}%'
    )
    ax.text(
        0.02, 0.98, f'Test samples: {sample_count:,} · Monte Carlo N={mc_samples:,}',
        transform=ax.transAxes, va='top', ha='left', fontsize=9,
        bbox={'facecolor': 'white', 'edgecolor': '0.8', 'alpha': 0.85},
    )
    ax.grid(alpha=0.22)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output_dir / f'calibration_plot.{suffix}', dpi=240, bbox_inches='tight')
    plt.close(fig)


def save_notes(output_dir, checkpoint_path, checkpoint, cfg, sample_count,
               mc_samples, times, curves, smoke):
    notes = f"""# Demo 3 — IMPTC M=3 calibration

- Dataset/split: IMPTC test
- Model: baseline LSTM–MDN, M=3
- Checkpoint: `{checkpoint_path}`
- Checkpoint epoch: {int(checkpoint['epoch'])}
- Checkpoint selection: minimum validation NLL
- Test samples: {sample_count}
- Monte Carlo samples per prediction: {mc_samples}
- Forecast times: {', '.join(f'{time:.1f}s' for time in times)}
- Overall Ravg: {curves['ravg_percent']:.6f}%
- Overall Rmin: {curves['rmin_percent']:.6f}%
- Development subset: {'yes — do not use on slides' if smoke else 'no — full test set'}

For each ground-truth point `p`, confidence is estimated exactly like the
repository evaluator: draw `N` samples `z` from the predicted GMM and compute
the fraction satisfying `D(z) > D(p)`. Calibration uses the repository's bins
`0.00, 0.01, ..., 1.00`; ideal behavior is observed frequency = expected
confidence. Curves below the diagonal indicate over-confidence.

The metric is computed over the full test split, not the eight fixed samples.
No synthetic GMM parameters are used.
"""
    (output_dir / 'demo3_notes.md').write_text(notes)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_global_seed(args.seed)
    run_dir, checkpoint_path, checkpoint, cfg, loader, model = load_inputs(args, device)
    output_dir = (args.output_dir or (run_dir / 'figures' / 'demo3')).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mc_samples = args.mc_samples or int(cfg.test_params['num_samples'])
    if mc_samples <= 0:
        raise ValueError('--mc-samples must be positive')
    confidence_sets = collect_confidence_sets(
        cfg, loader, model, device, mc_samples, args.max_test_samples
    )
    bins = np.asarray(cfg.reliability_bins, dtype=np.float64)
    times = np.asarray([
        (step + 1) * cfg.model_params['delta_t']
        for step in cfg.test_params['test_horizons']
    ])
    curves = calibration_curves(confidence_sets, bins)
    np.savez_compressed(
        output_dir / 'confidence_levels.npz',
        confidence_sets=confidence_sets.astype(np.float32), bins=bins,
        times_seconds=times, evaluation_seed=np.int64(args.seed),
        monte_carlo_samples=np.int64(mc_samples),
    )
    save_csvs(output_dir, bins, times, curves)
    plot_calibration(output_dir, bins, times, curves, len(confidence_sets), mc_samples)
    save_notes(
        output_dir, checkpoint_path, checkpoint, cfg, len(confidence_sets),
        mc_samples, times, curves, args.max_test_samples is not None,
    )
    summary = {
        'dataset': 'IMPTC', 'split': 'test', 'num_gaussians': 3,
        'checkpoint': str(checkpoint_path), 'checkpoint_epoch': int(checkpoint['epoch']),
        'sample_count': len(confidence_sets), 'monte_carlo_samples': mc_samples,
        'times_seconds': times.tolist(),
        'ravg_percent': curves['ravg_percent'], 'rmin_percent': curves['rmin_percent'],
        'ravg_by_horizon_percent': curves['ravg_by_horizon_percent'].tolist(),
        'rmin_by_horizon_percent': curves['rmin_by_horizon_percent'].tolist(),
        'full_test_set': args.max_test_samples is None,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Demo 3 written to: {output_dir}')
    print(f'Ravg={curves["ravg_percent"]:.2f}%  Rmin={curves["rmin_percent"]:.2f}%')


if __name__ == '__main__':
    main()
