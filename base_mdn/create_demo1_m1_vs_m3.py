#!/usr/bin/env python3
"""Create Demo 1: side-by-side M=1 vs M=3 confidence regions on one sample.

The argument of this demo is visual, not numerical: a single Gaussian must
stretch one ellipse across the gap between branches, while a mixture can place
mass on the branches and leave the gap empty.

No model is trained or run. Confidence regions reuse the Monte Carlo
highest-density-set construction and mesh-area convention of ``eval.py``, via
the helpers already validated in ``create_demo2_horizon_uncertainty.py``.

Sample selection follows EXPERIMENT_PROTOCOL.md section 15.2
(difficult/uncertain case): high mixture entropy, several components carrying
meaningful weight, and components that are actually separated in space. The
per-sample table is written to disk so the choice is reproducible and auditable
rather than picked by eye.
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

from create_demo2_horizon_uncertainty import (
    build_grid,
    confidence_map,
    contour_bounds,
    equal_limits,
    region_area,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_M1_RUN = PROJECT_ROOT / (
    'results/trained_models/base_mdn/imptc/m1_peds_imptc/runs/imptc_m1_seed2024'
)
DEFAULT_M3_RUN = PROJECT_ROOT / (
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
        description='Create the two-panel M=1 vs M=3 Demo 1 figure from saved artifacts.'
    )
    parser.add_argument('--m1-run', type=Path, default=DEFAULT_M1_RUN)
    parser.add_argument('--m3-run', type=Path, default=DEFAULT_M3_RUN)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--sample-id', default=None,
                        help='Fixed sample ID. By default pick the most multimodal case.')
    parser.add_argument('--time', type=float, default=None,
                        help='Forecast time in seconds. By default scan every horizon.')
    parser.add_argument('--min-weight', type=float, default=0.15,
                        help='Mixture weight above which a component counts as significant.')
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return parser.parse_args()


def load_run(run_dir, expected_m):
    run_dir = run_dir.expanduser().resolve()
    config_path = run_dir / 'resolved_config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing resolved config: {config_path}')
    config = json.loads(config_path.read_text())
    actual_m = int(config['model_params']['num_gaussians'])
    if actual_m != expected_m:
        raise ValueError(f'Expected M={expected_m} in {run_dir}, found M={actual_m}')
    inputs = np.load(run_dir / 'fixed_samples' / 'inputs.npz', allow_pickle=False)
    prediction = np.load(
        run_dir / 'fixed_samples' / 'predictions' / 'best.npz', allow_pickle=False
    )
    if prediction['pi'].shape[2] != expected_m:
        raise ValueError(
            f"Expected K={expected_m} prediction, got shape {prediction['pi'].shape}"
        )
    if not np.array_equal(inputs['sample_ids'], prediction['sample_ids']):
        raise ValueError(f'Fixed input and prediction sample IDs do not match in {run_dir}')
    return run_dir, config, inputs, prediction


def require_matching_runs(m1, m3):
    """Both runs must describe the same experiment apart from num_gaussians."""
    _, m1_config, m1_inputs, _ = m1
    _, m3_config, m3_inputs, _ = m3
    if not np.array_equal(m1_inputs['sample_ids'], m3_inputs['sample_ids']):
        raise ValueError('The two runs do not share the same fixed samples')
    if not np.allclose(m1_inputs['X'], m3_inputs['X']):
        raise ValueError('Fixed-sample inputs differ between the two runs')
    if not np.allclose(m1_inputs['y'], m3_inputs['y']):
        raise ValueError('Fixed-sample targets differ between the two runs')
    differences = []
    for key in sorted(set(m1_config['model_params']) | set(m3_config['model_params'])):
        left = m1_config['model_params'].get(key)
        right = m3_config['model_params'].get(key)
        if left != right:
            differences.append({'field': key, 'm1': left, 'm3': right})
    if [row['field'] for row in differences] != ['num_gaussians']:
        raise ValueError(f'Runs differ beyond num_gaussians: {differences}')
    return differences


def component_sigma(covariance):
    """Average standard deviation of a component, used as the length unit."""
    variances = np.clip(np.diagonal(covariance, axis1=-2, axis2=-1), 1e-12, None)
    return float(np.sqrt(variances.mean()))


def multimodality_score(pi, mu, covariance, min_weight):
    """Score one (sample, timestep) against EXPERIMENT_PROTOCOL.md 15.2.

    A case is interesting when several components carry real weight AND their
    means are far apart relative to their own spread. Weight alone is not
    enough: three components stacked on the same spot are not multimodal.
    """
    pi = np.asarray(pi, dtype=np.float64)
    safe = np.clip(pi, 1e-12, None)
    entropy = float(-np.sum(safe * np.log(safe)))
    perplexity = float(np.exp(entropy))          # effective number of components
    significant = int(np.sum(pi >= min_weight))

    sigma = float(np.mean([component_sigma(covariance[k]) for k in range(len(pi))]))
    separation_m = 0.0
    for i in range(len(pi)):
        for j in range(i + 1, len(pi)):
            if pi[i] < min_weight or pi[j] < min_weight:
                continue
            separation_m = max(separation_m, float(np.linalg.norm(mu[i] - mu[j])))
    separation_sigma = separation_m / max(sigma, 1e-9)

    # Components must be separated by more than 2 sigma before an equal-weight
    # mixture is genuinely bimodal, so that is the gate rather than a soft term.
    qualified = significant >= 2 and separation_sigma > 2.0
    score = perplexity * separation_sigma if qualified else 0.0
    return {
        'entropy': entropy,
        'perplexity': perplexity,
        'significant_components': significant,
        'max_separation_m': separation_m,
        'max_separation_sigma': separation_sigma,
        'mean_component_sigma_m': sigma,
        'qualified': bool(qualified),
        'score': float(score),
    }


def scan_candidates(inputs, prediction, time_index, min_weight):
    rows = []
    n_samples, horizon = prediction['pi'].shape[0], prediction['pi'].shape[1]
    steps = range(horizon) if time_index is None else [time_index]
    for sample_index in range(n_samples):
        best = None
        for timestep in steps:
            stats = multimodality_score(
                prediction['pi'][sample_index, timestep],
                prediction['mu'][sample_index, timestep],
                prediction['covariance'][sample_index, timestep],
                min_weight,
            )
            stats['timestep'] = int(timestep)
            if best is None or stats['score'] > best['score']:
                best = stats
        best['sample_index'] = int(sample_index)
        best['sample_id'] = str(inputs['sample_ids'][sample_index])
        rows.append(best)
    return rows


def choose_candidate(rows, requested_sample_id, time_index):
    if requested_sample_id is not None:
        matches = [row for row in rows if row['sample_id'] == requested_sample_id]
        if not matches:
            available = ', '.join(row['sample_id'] for row in rows)
            raise ValueError(
                f'Unknown fixed sample ID {requested_sample_id!r}. Available: {available}'
            )
        return matches[0], 'explicit --sample-id'
    selected = max(rows, key=lambda row: row['score'])
    if selected['score'] <= 0.0:
        raise SystemExit(
            'No fixed sample qualifies as multimodal: every candidate has fewer than two\n'
            'significant components or separation below 2 sigma. Demo 1 would show two\n'
            'nearly identical panels and would not make its point.\n'
            'Options: lower --min-weight, pass --sample-id explicitly, or draw Demo 1 on a\n'
            'synthetic branching dataset and label it as an illustration.'
        )
    reason = 'highest multimodality score (protocol 15.2)'
    if time_index is None:
        reason += ', horizon chosen per sample'
    return selected, reason


def panel_payload(prediction, sample_index, timestep, grid, grid_shape, mesh_area,
                  num_samples, seed, device):
    confidence = confidence_map(
        prediction['pi'][sample_index, timestep],
        prediction['mu'][sample_index, timestep],
        prediction['covariance'][sample_index, timestep],
        grid, num_samples, seed, device,
    ).reshape(grid_shape)
    a68 = region_area(confidence, 0.68, mesh_area)
    a95 = region_area(confidence, 0.95, mesh_area)
    if a68 > a95 + 1e-9:
        raise AssertionError('68% confidence region cannot exceed 95% region')
    return {
        'confidence': confidence,
        'a68_m2': a68,
        'a95_m2': a95,
        'pi': np.asarray(prediction['pi'][sample_index, timestep]).tolist(),
    }


def plot_demo(output_dir, inputs, sample_index, timestep, seconds, panels, xs, ys,
              selected):
    observed = inputs['X'][sample_index, :, :2]
    truth = inputs['y'][sample_index]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharex=True, sharey=True)
    for ax, (label, payload) in zip(axes, panels.items()):
        confidence = payload['confidence']
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
        ax.scatter(truth[timestep, 0], truth[timestep, 1], marker='*', s=135,
                   color=COLORS['ground_truth_point'], edgecolor=COLORS['truth'],
                   linewidth=0.8, zorder=8)
        weights = ', '.join(f'{value:.2f}' for value in payload['pi'])
        ax.set_title(f'{label}   (mixture weights: {weights})')
        ax.text(
            0.03, 0.03,
            f"A68 = {payload['a68_m2']:.2f} m²\nA95 = {payload['a95_m2']:.2f} m²",
            transform=ax.transAxes, fontsize=9.5, va='bottom', ha='left',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78, 'pad': 3},
        )
        ax.set_xlabel('X position [m]')
        ax.grid(alpha=0.2)
        ax.set_aspect('equal', adjustable='box')
    axes[0].set_ylabel('Y position [m]')

    region_points = []
    touches_boundary = False
    for payload in panels.values():
        points, touched = contour_bounds(xs, ys, [payload['confidence']])
        region_points.extend(points)
        touches_boundary |= touched
    # Identical limits on both panels; otherwise the area difference is not
    # readable from the figure.
    equal_limits(axes, [observed, truth, *region_points], padding=0.07)

    handles = [
        Line2D([0], [0], color=COLORS['past'], marker='o', label='Observed trajectory'),
        Line2D([0], [0], color=COLORS['truth'], marker='o', label='Ground-truth future'),
        Line2D([0], [0], marker='*', linestyle='None', markersize=11,
               markerfacecolor=COLORS['ground_truth_point'],
               markeredgecolor=COLORS['truth'], label='Ground truth at this horizon'),
        Line2D([0], [0], color=COLORS['region68'], linewidth=7, alpha=0.55,
               label='68% confidence region'),
        Line2D([0], [0], color=COLORS['region95'], linewidth=7, alpha=0.55,
               label='95% confidence region'),
    ]
    a95_m1 = panels['M = 1 (single Gaussian)']['a95_m2']
    a95_m3 = panels['M = 3 (mixture)']['a95_m2']
    ratio = a95_m1 / a95_m3 if a95_m3 > 0 else float('nan')
    fig.suptitle(
        f'Single Gaussian vs mixture at t = +{seconds:.1f} s\n'
        f"sample {selected['sample_id']} · "
        f"{selected['significant_components']} significant components · "
        f"separation {selected['max_separation_sigma']:.1f} sigma · "
        f'A95 ratio M1/M3 = {ratio:.2f}',
        fontsize=12,
    )
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output_dir / f'demo1_m1_vs_m3.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return touches_boundary, ratio


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    m1 = load_run(args.m1_run, 1)
    m3 = load_run(args.m3_run, 3)
    differences = require_matching_runs(m1, m3)
    m1_run, m1_config, inputs, m1_prediction = m1
    m3_run, m3_config, _, m3_prediction = m3

    model_params = m3_config['model_params']
    test_params = m3_config['test_params']
    delta_t = float(model_params['delta_t'])
    horizon = int(model_params['forecast_horizon'])

    time_index = None
    if args.time is not None:
        step = int(round(args.time / delta_t))
        if not np.isclose(step * delta_t, args.time, atol=1e-8):
            raise ValueError(f'Time {args.time}s is not aligned with delta_t={delta_t}s')
        time_index = step - 1
        if time_index < 0 or time_index >= horizon:
            raise ValueError(f'Time {args.time}s lies outside the forecast horizon')

    # Candidate ranking uses the mixture run: M=1 has nothing to be multimodal about.
    rows = scan_candidates(inputs, m3_prediction, time_index, args.min_weight)
    selected, reason = choose_candidate(rows, args.sample_id, time_index)
    sample_index = selected['sample_index']
    timestep = selected['timestep']
    seconds = (timestep + 1) * delta_t

    xs, ys, grid = build_grid(
        float(test_params['mesh_range_x']), float(test_params['mesh_range_y']),
        float(test_params['mesh_resolution']), device,
    )
    grid_shape = (len(ys), len(xs))
    mesh_area = float(test_params['mesh_range_x']) * float(test_params['mesh_range_y'])

    panels = {}
    for label, prediction in (
        ('M = 1 (single Gaussian)', m1_prediction),
        ('M = 3 (mixture)', m3_prediction),
    ):
        # Same seed for both panels so the Monte Carlo stream cannot explain
        # any difference between them.
        panels[label] = panel_payload(
            prediction, sample_index, timestep, grid, grid_shape, mesh_area,
            args.num_samples, args.seed, device,
        )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'results/comparisons/imptc_m1_vs_m3/demo1'
    output_dir = output_dir.expanduser().resolve()
    touches_boundary, ratio = plot_demo(
        output_dir, inputs, sample_index, timestep, seconds, panels, xs, ys, selected,
    )

    report = {
        'schema_version': '1.0',
        'dataset': 'IMPTC',
        'source': 'fixed-sample predictions at the validation-best checkpoint',
        'm1_run': str(m1_run),
        'm3_run': str(m3_run),
        'config_differences': differences,
        'selection_rule': 'EXPERIMENT_PROTOCOL.md 15.2 difficult/uncertain case',
        'selection_reason': reason,
        'min_weight': args.min_weight,
        'separation_gate_sigma': 2.0,
        'selected': {key: value for key, value in selected.items()},
        'selected_time_seconds': seconds,
        'monte_carlo_samples': args.num_samples,
        'monte_carlo_seed': args.seed,
        'a68_m2': {label: payload['a68_m2'] for label, payload in panels.items()},
        'a95_m2': {label: payload['a95_m2'] for label, payload in panels.items()},
        'a95_ratio_m1_over_m3': ratio,
        'confidence_region_touches_mesh_boundary': bool(touches_boundary),
        'candidates': rows,
    }
    (output_dir / 'demo1_selection.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    write_csv(output_dir / 'demo1_candidates.csv', rows)

    print(json.dumps({key: report[key] for key in (
        'selection_reason', 'selected_time_seconds', 'a68_m2', 'a95_m2',
        'a95_ratio_m1_over_m3', 'confidence_region_touches_mesh_boundary',
    )}, indent=2, ensure_ascii=False))
    print(f"Selected sample: {selected['sample_id']} "
          f"(score {selected['score']:.2f}, "
          f"{selected['significant_components']} significant components, "
          f"separation {selected['max_separation_sigma']:.2f} sigma)")
    if touches_boundary:
        print('WARNING: a confidence region touches the mesh boundary; the reported '
              'area is a lower bound. Increase mesh_range_x/mesh_range_y to fix.')
    if ratio < 1.05:
        print('WARNING: the M=1 region is not visibly larger than M=3 on this sample. '
              'Demo 1 will look unconvincing. Try another --sample-id or --time.')
    print(f'Demo 1 artifacts: {output_dir}')


if __name__ == '__main__':
    main()
