#!/usr/bin/env python3
"""Create Demo 2 from saved, real M=3 fixed-sample predictions.

No model is trained or run. Confidence regions use the same Monte Carlo
highest-density-set construction and mesh-area convention as ``eval.py``.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.distributions as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT_ROOT / (
    'results/trained_models/base_mdn/imptc/default_peds_imptc/'
    'runs/imptc_baseline_seed2024'
)
COLORS = {
    'past': '#D55E00',
    'truth': '#111111',
    'ground_truth_point': '#F0E442',
    'region68': '#0072B2',
    'region95': '#56B4E9',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create four-panel GMM confidence-region Demo 2 from saved artifacts.'
    )
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--sample-id', default=None,
                        help='Fixed sample ID. By default select the strongest actual A95 expansion.')
    parser.add_argument('--times', type=float, nargs=4, default=[1.0, 2.0, 3.0, 4.8])
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return parser.parse_args()


def load_artifacts(run_dir):
    run_dir = run_dir.expanduser().resolve()
    config = json.loads((run_dir / 'resolved_config.json').read_text())
    inputs = np.load(run_dir / 'fixed_samples' / 'inputs.npz', allow_pickle=False)
    prediction = np.load(
        run_dir / 'fixed_samples' / 'predictions' / 'best.npz', allow_pickle=False
    )
    if int(config['model_params']['num_gaussians']) != 3:
        raise ValueError('Demo 2 requires the trained M=3 run')
    if prediction['pi'].shape[2] != 3:
        raise ValueError(f"Expected K=3 prediction, got shape {prediction['pi'].shape}")
    if not np.array_equal(inputs['sample_ids'], prediction['sample_ids']):
        raise ValueError('Fixed input and prediction sample IDs do not match')
    return run_dir, config, inputs, prediction


def time_indices(times, delta_t, forecast_horizon):
    indices = []
    for seconds in times:
        step = int(round(seconds / delta_t))
        if not np.isclose(step * delta_t, seconds, atol=1e-8):
            raise ValueError(f'Time {seconds}s is not aligned with delta_t={delta_t}s')
        index = step - 1
        if index < 0 or index >= forecast_horizon:
            raise ValueError(f'Time {seconds}s lies outside the forecast horizon')
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError('Selected times map to duplicate forecast indices')
    return indices


def build_grid(mesh_range_x, mesh_range_y, mesh_resolution, device):
    # This intentionally mirrors MDN_Forecaster.build_mesh_grid.
    steps = int(((mesh_range_x + mesh_range_y) / mesh_resolution) + 1)
    xs = torch.linspace(-mesh_range_x, mesh_range_x, steps=steps, device=device)
    ys = torch.linspace(-mesh_range_y, mesh_range_y, steps=steps, device=device)
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing='xy')
    points = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
    return xs.cpu().numpy(), ys.cpu().numpy(), points


def confidence_map(pi, mu, covariance, grid, num_samples, seed, device):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    pi_tensor = torch.as_tensor(pi, dtype=torch.float32, device=device)
    mu_tensor = torch.as_tensor(mu, dtype=torch.float32, device=device)
    covariance_tensor = torch.as_tensor(covariance, dtype=torch.float32, device=device)
    mixture = dist.MixtureSameFamily(
        dist.Categorical(probs=pi_tensor),
        dist.MultivariateNormal(mu_tensor, covariance_tensor),
    )
    sample_log_prob = mixture.log_prob(mixture.sample((num_samples,)))
    grid_log_prob = mixture.log_prob(grid)
    # Same definition as eval.py: fraction of model samples with greater
    # density than each queried grid point.
    confidence = (sample_log_prob[:, None] > grid_log_prob[None, :]).float().mean(dim=0)
    return confidence.cpu().numpy()


def region_area(confidence, level, mesh_area):
    return float(np.mean(confidence <= level) * mesh_area)


def compute_candidate(inputs, prediction, sample_index, indices, grid, grid_shape,
                      mesh_area, num_samples, seed, device):
    rows = []
    maps = []
    for position, timestep in enumerate(indices):
        confidence = confidence_map(
            prediction['pi'][sample_index, timestep],
            prediction['mu'][sample_index, timestep],
            prediction['covariance'][sample_index, timestep],
            grid, num_samples, seed + sample_index * 100 + position, device,
        ).reshape(grid_shape)
        a68 = region_area(confidence, 0.68, mesh_area)
        a95 = region_area(confidence, 0.95, mesh_area)
        if a68 > a95 + 1e-9:
            raise AssertionError('68% confidence region cannot exceed 95% region')
        rows.append({'timestep': timestep, 'a68_m2': a68, 'a95_m2': a95})
        maps.append(confidence)
    a95 = np.asarray([row['a95_m2'] for row in rows])
    differences = np.diff(a95)
    observed = inputs['X'][sample_index, :, :2]
    truth = inputs['y'][sample_index]
    return {
        'sample_index': sample_index,
        'sample_id': str(inputs['sample_ids'][sample_index]),
        'rows': rows,
        'maps': maps,
        'net_a95_expansion_m2': float(a95[-1] - a95[0]),
        'monotonic_transitions': int(np.sum(differences >= 0.0)),
        'strictly_monotonic': bool(np.all(differences >= 0.0)),
        'observed_path_length_m': float(np.linalg.norm(np.diff(observed, axis=0), axis=1).sum()),
        'future_path_length_m': float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum()),
        'future_displacement_m': float(np.linalg.norm(truth[-1] - observed[-1])),
    }


def choose_candidate(candidates, requested_sample_id):
    if requested_sample_id is not None:
        matches = [row for row in candidates if row['sample_id'] == requested_sample_id]
        if not matches:
            available = ', '.join(row['sample_id'] for row in candidates)
            raise ValueError(f'Unknown fixed sample ID {requested_sample_id!r}. Available: {available}')
        return matches[0], 'explicit --sample-id'
    # Prefer actual monotonic examples with a visible future trajectory, then
    # the largest measured end-to-start expansion. This avoids selecting a
    # nearly stationary person whose trajectory context collapses to one point.
    selected = max(
        candidates,
        key=lambda row: (
            row['strictly_monotonic'], row['monotonic_transitions'],
            row['future_displacement_m'], row['net_a95_expansion_m2'],
        ),
    )
    return selected, 'prefer monotonic A95 change, then largest visible future displacement among 8 fixed samples'


def contour_bounds(xs, ys, confidence_maps, level=0.95):
    points = []
    touches_boundary = False
    for confidence in confidence_maps:
        mask = confidence <= level
        if not np.any(mask):
            continue
        row_indices, column_indices = np.where(mask)
        touches_boundary |= bool(
            np.any(row_indices == 0) or np.any(row_indices == len(ys) - 1)
            or np.any(column_indices == 0) or np.any(column_indices == len(xs) - 1)
        )
        points.append(np.column_stack([xs[column_indices], ys[row_indices]]))
    return points, touches_boundary


def equal_limits(axes, point_sets, padding=0.08):
    points = np.concatenate([np.asarray(value).reshape(-1, 2) for value in point_sets], axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    span = max(float(np.max(maximum - minimum)), 1e-3) * (1.0 + 2.0 * padding)
    for ax in axes:
        ax.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
        ax.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
        ax.set_aspect('equal', adjustable='box')


def plot_demo(output_dir, inputs, selected, indices, times, xs, ys):
    sample_index = selected['sample_index']
    observed = inputs['X'][sample_index, :, :2]
    truth = inputs['y'][sample_index]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharex=True, sharey=True)
    for ax, timestep, seconds, confidence, area_row in zip(
            axes, indices, times, selected['maps'], selected['rows']):
        ax.contourf(
            xs, ys, confidence, levels=[0.0, 0.68, 0.95],
            colors=[COLORS['region68'], COLORS['region95']], alpha=0.42,
        )
        ax.contour(
            xs, ys, confidence, levels=[0.68, 0.95],
            colors=[COLORS['region68'], COLORS['region95']], linewidths=[1.8, 1.5],
        )
        ax.plot(observed[:, 0], observed[:, 1], '-o', color=COLORS['past'],
                linewidth=2.0, markersize=2.5, zorder=5)
        ax.scatter(observed[-1, 0], observed[-1, 1], marker='D', s=58,
                   color=COLORS['past'], edgecolor='white', linewidth=0.7, zorder=7)
        ax.plot(truth[:, 0], truth[:, 1], '-o', color=COLORS['truth'],
                linewidth=1.7, markersize=2.1, zorder=5)
        ax.scatter(truth[timestep, 0], truth[timestep, 1], marker='*', s=125,
                   color=COLORS['ground_truth_point'], edgecolor=COLORS['truth'],
                   linewidth=0.8, zorder=8)
        ax.set_title(f't = +{seconds:.1f} s')
        ax.text(
            0.03, 0.03,
            f"A68 = {area_row['a68_m2']:.2f} m²\nA95 = {area_row['a95_m2']:.2f} m²",
            transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78, 'pad': 3},
        )
        ax.set_xlabel('X position [m]')
        ax.grid(alpha=0.2)
        ax.set_aspect('equal', adjustable='box')
    axes[0].set_ylabel('Y position [m]')

    region_points, touches_boundary = contour_bounds(xs, ys, selected['maps'])
    equal_limits(axes, [observed, truth, *region_points], padding=0.07)
    handles = [
        Line2D([0], [0], color=COLORS['past'], marker='o', label='Observed trajectory'),
        Line2D([0], [0], color=COLORS['truth'], marker='o', label='Ground-truth future'),
        Line2D([0], [0], marker='*', linestyle='None', markersize=11,
               markerfacecolor=COLORS['ground_truth_point'], markeredgecolor=COLORS['truth'],
               label='Ground truth at selected horizon'),
        Line2D([0], [0], color=COLORS['region68'], linewidth=7, alpha=0.55,
               label='68% GMM confidence region'),
        Line2D([0], [0], color=COLORS['region95'], linewidth=7, alpha=0.55,
               label='95% GMM confidence region'),
    ]
    trend = 'monotonic expansion' if selected['strictly_monotonic'] else 'non-monotonic change'
    fig.suptitle(
        f'GMM confidence regions across the forecast horizon\n'
        f'Actual M=3 best-checkpoint output · {trend}', fontsize=14,
    )
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 0.91), w_pad=1.0)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output_dir / f'demo2_horizon_uncertainty.{suffix}', dpi=240,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return touches_boundary


def write_csv(path, candidates, times):
    with path.open('w', newline='') as stream:
        fields = ['sample_index', 'sample_id', 'time_seconds', 'future_index',
                  'a68_m2', 'a95_m2', 'net_a95_expansion_m2',
                  'monotonic_transitions', 'strictly_monotonic',
                  'observed_path_length_m', 'future_path_length_m', 'future_displacement_m']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            for seconds, row in zip(times, candidate['rows']):
                writer.writerow({
                    'sample_index': candidate['sample_index'],
                    'sample_id': candidate['sample_id'],
                    'time_seconds': seconds,
                    'future_index': row['timestep'],
                    'a68_m2': row['a68_m2'],
                    'a95_m2': row['a95_m2'],
                    'net_a95_expansion_m2': candidate['net_a95_expansion_m2'],
                    'monotonic_transitions': candidate['monotonic_transitions'],
                    'strictly_monotonic': candidate['strictly_monotonic'],
                    'observed_path_length_m': candidate['observed_path_length_m'],
                    'future_path_length_m': candidate['future_path_length_m'],
                    'future_displacement_m': candidate['future_displacement_m'],
                })


def main():
    args = parse_args()
    run_dir, config, inputs, prediction = load_artifacts(args.run_dir)
    output_dir = (args.output_dir or (run_dir / 'figures' / 'demo2')).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model_params = config['model_params']
    test_params = config['test_params']
    delta_t = float(model_params['delta_t'])
    indices = time_indices(args.times, delta_t, int(model_params['forecast_horizon']))
    mesh_range_x = float(test_params['mesh_range_x'])
    mesh_range_y = float(test_params['mesh_range_y'])
    mesh_resolution = float(test_params['mesh_resolution'])
    xs, ys, grid = build_grid(mesh_range_x, mesh_range_y, mesh_resolution, device)
    mesh_area = (2.0 * mesh_range_x) * (2.0 * mesh_range_y)
    grid_shape = (len(ys), len(xs))

    candidates = [
        compute_candidate(
            inputs, prediction, sample_index, indices, grid, grid_shape,
            mesh_area, args.num_samples, args.seed, device,
        )
        for sample_index in range(len(inputs['sample_ids']))
    ]
    selected, selection_rule = choose_candidate(candidates, args.sample_id)
    touches_boundary = plot_demo(output_dir, inputs, selected, indices, args.times, xs, ys)
    write_csv(output_dir / 'demo2_all_fixed_sample_areas.csv', candidates, args.times)

    checkpoint_epoch = int(np.asarray(prediction['epoch']).item())
    notes = [
        '# Demo 2 — Confidence regions across forecast horizon', '',
        f'- Run: `{run_dir}`',
        f'- Checkpoint: `best.pt`, epoch `{checkpoint_epoch}`',
        f"- Sample ID: `{selected['sample_id']}` (fixed index `{selected['sample_index']}`)",
        f'- Selection rule: {selection_rule}.',
        '- Selection is for illustration only; quantitative conclusions use the full test set.',
        f'- Future times: `{args.times}` seconds; zero-based indices: `{indices}`.',
        '- Model: `M=3`; confidence regions use the complete mixture, not one component.',
        f'- Monte Carlo samples per distribution: `{args.num_samples}`; seed: `{args.seed}`.',
        f'- Mesh: x/y half-range `{mesh_range_x:g}/{mesh_range_y:g}` m, resolution `{mesh_resolution:g}` m.',
        '- Region construction matches `eval.py`: confidence at a grid point is the fraction of GMM samples with greater density; the κ-region uses `confidence <= κ`.',
        '- `A68(t)` and `A95(t)` are single-sample, single-horizon grid areas in m². They are not the aggregate `S68`/`S95` sharpness scores.',
        f"- Observed A95 trend: {'monotonic expansion' if selected['strictly_monotonic'] else 'non-monotonic change'}; net change `{selected['net_a95_expansion_m2']:.4f}` m².",
        f"- Future displacement from last observation: `{selected['future_displacement_m']:.4f}` m.",
        f"- Confidence region touches mesh boundary: `{touches_boundary}`.", '',
        '| Time | Future index | A68 [m²] | A95 [m²] |',
        '|---:|---:|---:|---:|',
    ]
    for seconds, row in zip(args.times, selected['rows']):
        notes.append(
            f"| {seconds:.1f} s | {row['timestep']} | {row['a68_m2']:.4f} | {row['a95_m2']:.4f} |"
        )
    (output_dir / 'demo2_horizon_uncertainty_notes.md').write_text('\n'.join(notes) + '\n')
    summary = {
        'sample_id': selected['sample_id'],
        'sample_index': selected['sample_index'],
        'selection_rule': selection_rule,
        'times_seconds': args.times,
        'future_indices_zero_based': indices,
        'areas': selected['rows'],
        'net_a95_expansion_m2': selected['net_a95_expansion_m2'],
        'strictly_monotonic': selected['strictly_monotonic'],
        'observed_path_length_m': selected['observed_path_length_m'],
        'future_path_length_m': selected['future_path_length_m'],
        'future_displacement_m': selected['future_displacement_m'],
        'touches_mesh_boundary': touches_boundary,
    }
    (output_dir / 'demo2_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Demo 2 artifacts: {output_dir}')


if __name__ == '__main__':
    main()
