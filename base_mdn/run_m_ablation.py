#!/usr/bin/env python3
"""Evaluate every trained M in the ablation and plot the resulting curves.

The paper states ``we set the number of Gaussians to three`` for all datasets,
but the official configs use M=3 for IMPTC and M=5 for ETH/UCY, and no ablation
is reported. This script measures the quantity the paper leaves untested.

Evaluation reuses ``compare_m1_m3.py`` so every model goes through the same
official metric path, with the evaluation seed reset before each model.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_m1_m3 import (
    build_config,
    config_differences,
    evaluate_model,
    json_safe,
    read_json,
    validate_run,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / 'base_mdn/configs/imptc'
RESULT_ROOT = PROJECT_ROOT / 'results/trained_models/base_mdn/imptc'

# M -> (config file, run directory). M=3 is the untouched official baseline.
SWEEP = {
    1: ('m1_peds_imptc.json', 'm1_peds_imptc/runs/imptc_m1_seed2024'),
    2: ('m2_peds_imptc.json', 'm2_peds_imptc/runs/imptc_m2_seed2024'),
    3: ('default_peds_imptc.json', 'default_peds_imptc/runs/imptc_baseline_seed2024'),
    5: ('m5_peds_imptc.json', 'm5_peds_imptc/runs/imptc_m5_seed2024'),
    8: ('m8_peds_imptc.json', 'm8_peds_imptc/runs/imptc_m8_seed2024'),
}
PAPER_CONFIG_M = 3          # what the official IMPTC config ships with


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the num_gaussians ablation and plot reliability vs M.'
    )
    parser.add_argument('--only', type=int, nargs='+', default=None,
                        help='Subset of M values to evaluate, e.g. --only 1 3.')
    parser.add_argument('--output-dir', type=Path,
                        default=PROJECT_ROOT / 'results/ablations/imptc_num_gaussians')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--evaluation-seed', type=int, default=2024)
    parser.add_argument('--skip-missing', action='store_true',
                        help='Skip M values whose run is absent instead of failing.')
    return parser.parse_args()


def check_only_num_gaussians_differs(results_configs):
    """Every run must be the same experiment apart from num_gaussians."""
    items = sorted(results_configs.items())
    reference_m, reference_config = items[0]
    problems = []
    for m, config in items[1:]:
        differences = config_differences(reference_config, config)
        unexpected = [
            row['field'] for row in differences
            if row['field'] != 'model_params.num_gaussians'
        ]
        if unexpected:
            problems.append({'m': m, 'against_m': reference_m, 'fields': unexpected})
    if problems:
        raise ValueError(
            'Ablation runs differ beyond num_gaussians, so the comparison would not be '
            f'controlled: {problems}'
        )


def plot_ablation(path, results):
    ms = [row['num_gaussians'] for row in results]
    r_avg = [row['ravg_percent'] for row in results]
    r_min = [row['rmin_percent'] for row in results]
    s95 = [row['s95_m2_per_s'] for row in results]
    ade = [row['minade20_m'] for row in results]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(9, 8.5), sharex=True,
        gridspec_kw={'height_ratios': [1.35, 1.0]},
    )

    ax_top.plot(ms, r_avg, '-o', color='#0072B2', linewidth=2.2, label='R_avg')
    ax_top.plot(ms, r_min, '-s', color='#D55E00', linewidth=2.2, label='R_min')
    ax_top.axhline(95.0, color='#0072B2', linestyle=':', linewidth=1.3,
                   label='R_avg threshold 95%')
    ax_top.axhline(90.0, color='#D55E00', linestyle=':', linewidth=1.3,
                   label='R_min threshold 90%')
    if PAPER_CONFIG_M in ms:
        ax_top.axvline(PAPER_CONFIG_M, color='#444444', linestyle='--', linewidth=1.2)
        ax_top.annotate('official IMPTC config', xy=(PAPER_CONFIG_M, min(r_min)),
                        xytext=(4, 6), textcoords='offset points', fontsize=9,
                        rotation=90, color='#444444')
    for x, y in zip(ms, r_min):
        ax_top.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                        xytext=(0, -14), ha='center', fontsize=8.5, color='#D55E00')
    ax_top.set_ylabel('Reliability [%]  (higher is better)')
    ax_top.grid(alpha=0.25)
    ax_top.legend(fontsize=9)
    ax_top.set_title('IMPTC test set: does the number of mixture components matter?')

    ax_bottom.plot(ms, s95, '-^', color='#009E73', linewidth=2.2, label='S95 [m²/s]')
    ax_bottom.set_ylabel('S95 [m²/s]  (lower is better)', color='#009E73')
    ax_bottom.tick_params(axis='y', labelcolor='#009E73')
    ax_bottom.grid(alpha=0.25)

    ax_ade = ax_bottom.twinx()
    ax_ade.plot(ms, ade, '-v', color='#CC79A7', linewidth=2.2, label='minADE20 [m]')
    ax_ade.set_ylabel('minADE20 [m]  (lower is better)', color='#CC79A7')
    ax_ade.tick_params(axis='y', labelcolor='#CC79A7')

    handles = ax_bottom.get_legend_handles_labels()[0] + ax_ade.get_legend_handles_labels()[0]
    labels = ax_bottom.get_legend_handles_labels()[1] + ax_ade.get_legend_handles_labels()[1]
    ax_bottom.legend(handles, labels, fontsize=9, loc='best')
    ax_bottom.set_xlabel('Number of mixture components M')
    ax_bottom.set_xticks(ms)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(path.with_suffix('.' + suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wanted = sorted(args.only) if args.only else sorted(SWEEP)
    unknown = [m for m in wanted if m not in SWEEP]
    if unknown:
        raise ValueError(f'No config registered for M={unknown}. Known: {sorted(SWEEP)}')

    results = []
    configs = {}
    skipped = []
    for m in wanted:
        config_name, run_suffix = SWEEP[m]
        run_dir = RESULT_ROOT / run_suffix
        if not (run_dir / 'checkpoints' / 'best.pt').is_file():
            if args.skip_missing:
                skipped.append({'m': m, 'run_dir': str(run_dir), 'reason': 'no best.pt'})
                print(f'[skip] M={m}: no trained run at {run_dir}')
                continue
            raise FileNotFoundError(
                f'M={m} has no trained run at {run_dir}. Train it first, or pass '
                f'--skip-missing to plot a partial curve.'
            )
        _, manifest, checkpoint, checkpoint_path = validate_run(run_dir, m)
        configs[m] = checkpoint['resolved_config']
        result = evaluate_model(
            f'm{m}', CONFIG_DIR / config_name, checkpoint, checkpoint_path,
            output_dir, args.evaluation_seed, device,
        )
        result['best_epoch'] = int(manifest['best_epoch'])
        result['run_dir'] = str(run_dir)
        results.append(result)
        print(f"[done] M={m}: R_avg={result['ravg_percent']:.1f}% "
              f"R_min={result['rmin_percent']:.1f}% "
              f"S95={result['s95_m2_per_s']:.2f} "
              f"minADE20={result['minade20_m']:.3f}")

    if len(results) < 2:
        raise SystemExit('Need at least two trained M values to draw an ablation curve.')
    check_only_num_gaussians_differs(configs)
    results.sort(key=lambda row: row['num_gaussians'])

    report = {
        'schema_version': '1.0',
        'dataset': 'IMPTC',
        'split': 'test',
        'checkpoint_selection': 'minimum validation NLL',
        'evaluation_seed': args.evaluation_seed,
        'device': str(device),
        'official_config_num_gaussians': PAPER_CONFIG_M,
        'paper_claim': 'Sec. 4.1 p.9 states M=3 for all datasets; the official ETH/UCY '
                       'config uses M=5, and no ablation is reported.',
        'evaluated': [row['num_gaussians'] for row in results],
        'skipped': skipped,
        'results': results,
    }
    report = json.loads(json.dumps(report, default=json_safe))
    (output_dir / 'ablation.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    write_csv(output_dir / 'ablation.csv', results)
    plot_ablation(output_dir / 'num_gaussians_ablation', results)

    best_rmin = max(results, key=lambda row: row['rmin_percent'])
    print()
    print(f"Best R_min at M={best_rmin['num_gaussians']} "
          f"({best_rmin['rmin_percent']:.1f}%); official config uses M={PAPER_CONFIG_M}.")
    print('Reminder: one seed per configuration, so read the trend, not the exact numbers.')
    print(f'Ablation artifacts: {output_dir}')


if __name__ == '__main__':
    main()
