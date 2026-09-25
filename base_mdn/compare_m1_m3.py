#!/usr/bin/env python3
"""Fairly compare validation-best M=1 and M=3 models on the IMPTC test set."""

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from base_lstm import LSTM_Trajectory_Forecast, NLL_MDN_loss
from eval import MDN_Forecaster
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.experiment import set_global_seed
from utils.helper import count_model_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_M1_RUN = PROJECT_ROOT / (
    'results/trained_models/base_mdn/imptc/m1_peds_imptc/'
    'runs/imptc_m1_seed2024'
)
DEFAULT_M3_RUN = PROJECT_ROOT / (
    'results/trained_models/base_mdn/imptc/default_peds_imptc/'
    'runs/imptc_baseline_seed2024'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare validation-best M=1 and M=3 checkpoints on full IMPTC test data.'
    )
    parser.add_argument('--m1-run', type=Path, default=DEFAULT_M1_RUN)
    parser.add_argument('--m3-run', type=Path, default=DEFAULT_M3_RUN)
    parser.add_argument(
        '--output-dir', type=Path,
        default=PROJECT_ROOT / 'results/comparisons/imptc_m1_vs_m3',
    )
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--evaluation-seed', type=int, default=2024)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text())


def flatten(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f'{prefix}.{key}' if prefix else key
            result.update(flatten(child, child_prefix))
    else:
        result[prefix] = value
    return result


def config_differences(m1_config, m3_config):
    left = flatten(m1_config)
    right = flatten(m3_config)
    differences = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            differences.append({'field': key, 'm1': left.get(key), 'm3': right.get(key)})
    return differences


def validate_run(run_dir, expected_m):
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / 'run_manifest.json'
    checkpoint_path = run_dir / 'checkpoints' / 'best.pt'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing run manifest: {manifest_path}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Missing validation-best checkpoint: {checkpoint_path}')
    manifest = read_json(manifest_path)
    if manifest.get('status') != 'completed':
        raise ValueError(f'Run is not completed: {run_dir}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_params = checkpoint.get('resolved_config', {}).get('model_params', {})
    actual_m = model_params.get('num_gaussians')
    if actual_m != expected_m:
        raise ValueError(f'Expected M={expected_m}, found M={actual_m}: {checkpoint_path}')
    if int(checkpoint['epoch']) != int(manifest['best_epoch']):
        raise ValueError(f'best.pt epoch does not match run manifest: {run_dir}')
    return run_dir, manifest, checkpoint, checkpoint_path


def build_config(config_path, output_dir):
    cfg = ConfigLoader(
        config_path=str(config_path), target='imptc', with_log=False, with_print=True,
        name=config_path.stem, model_arch='base_mdn', type='testing',
    )
    cfg.testing_path = str(output_dir)
    Path(cfg.testing_path).mkdir(parents=True, exist_ok=True)
    return cfg


def full_test_nll(model, data_loader, cfg, device):
    data_x, data_y, _, _, _ = data_loader.get_test_data()
    batch_size = int(cfg.test_params['batch_size'])
    input_horizons = int(cfg.test_params['num_input_horizons'])
    total_nll = 0.0
    total_values = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data_x), batch_size):
            inputs = torch.as_tensor(
                data_x[start:start + batch_size, -input_horizons:, :],
                dtype=torch.float32, device=device,
            )
            targets = torch.as_tensor(
                data_y[start:start + batch_size], dtype=torch.float32, device=device,
            )
            outputs = model(inputs)
            loss, diverged = NLL_MDN_loss(
                outputs, targets, cfg.model_params['num_gaussians']
            )
            if diverged or loss is None or not torch.isfinite(loss):
                raise RuntimeError('Non-finite test NLL')
            values = targets.shape[0] * targets.shape[1]
            total_nll += float(loss) * values
            total_values += values
    return total_nll / total_values


def evaluate_model(label, config_path, checkpoint, checkpoint_path, output_dir,
                   evaluation_seed, device):
    cfg = build_config(config_path, output_dir / label)
    saved_params = checkpoint['resolved_config']['model_params']
    if saved_params != cfg.model_params:
        raise ValueError(f'{label} checkpoint model_params do not match {config_path}')

    loader = DataLoader(cfg=cfg)
    loader.load_test_data()
    model = LSTM_Trajectory_Forecast(cfg=cfg.model_params).to(device)
    model.load_state_dict(checkpoint['model'])

    test_nll = full_test_nll(model, loader, cfg, device)
    # Reset before each model so both official stochastic evaluations use the
    # same pseudo-random stream and identical Monte Carlo settings.
    set_global_seed(evaluation_seed)
    forecaster = MDN_Forecaster(
        cfg=cfg, model=model, data_loader=loader, type='testing',
        logger=logging.getLogger(f'compare_{label}'), device=device,
    )
    official = forecaster.evaluate(epoch=None)
    return {
        'label': label,
        'num_gaussians': int(cfg.model_params['num_gaussians']),
        'checkpoint': str(checkpoint_path),
        'checkpoint_epoch': int(checkpoint['epoch']),
        'validation_nll': float(checkpoint['best_validation_nll']),
        'test_nll': float(test_nll),
        'parameter_count': int(count_model_parameters(model)),
        **official,
    }


def json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_csv(path, rows):
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def plot_summary(path, results):
    labels = [row['label'].upper() for row in results]
    groups = [
        ('NLL (lower is better)', ['validation_nll', 'test_nll']),
        ('Displacement (m, lower is better)', ['minade20_m', 'minfde20_m']),
        ('Reliability score (%, higher is better)', ['ravg_percent', 'rmin_percent']),
        ('Sharpness (m²/s, compare after reliability)', ['s68_m2_per_s', 's95_m2_per_s']),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ['#0072B2', '#D55E00']
    for ax, (title, metrics) in zip(axes.flat, groups):
        x = np.arange(len(metrics))
        width = 0.34
        for model_index, (label, row) in enumerate(zip(labels, results)):
            values = [row[key] for key in metrics]
            bars = ax.bar(x + (model_index - 0.5) * width, values, width,
                          label=label, color=colors[model_index])
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
        ax.set_xticks(x, metrics)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle('IMPTC test comparison: validation-best M=1 vs M=3')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    m1_run, m1_manifest, m1_checkpoint, m1_path = validate_run(args.m1_run, 1)
    m3_run, m3_manifest, m3_checkpoint, m3_path = validate_run(args.m3_run, 3)
    m1_config = m1_checkpoint['resolved_config']
    m3_config = m3_checkpoint['resolved_config']
    differences = config_differences(m1_config, m3_config)
    allowed = {'model_params.num_gaussians'}
    unexpected = [row for row in differences if row['field'] not in allowed]
    if unexpected:
        raise ValueError(f'Runs differ beyond num_gaussians: {unexpected}')
    if {row['field'] for row in differences} != allowed:
        raise ValueError(f'Expected exactly one config difference ({allowed}), got {differences}')

    config_dir = PROJECT_ROOT / 'base_mdn/configs/imptc'
    m1_result = evaluate_model(
        'm1', config_dir / 'm1_peds_imptc.json', m1_checkpoint, m1_path,
        output_dir, args.evaluation_seed, device,
    )
    m3_result = evaluate_model(
        'm3', config_dir / 'default_peds_imptc.json', m3_checkpoint, m3_path,
        output_dir, args.evaluation_seed, device,
    )
    results = [m1_result, m3_result]
    scalar_keys = sorted(
        key for key in set(m1_result) & set(m3_result)
        if isinstance(m1_result[key], (int, float))
        and isinstance(m3_result[key], (int, float))
        and key not in {'num_gaussians'}
    )
    delta = {key: m3_result[key] - m1_result[key] for key in scalar_keys}
    report = {
        'schema_version': '1.0',
        'dataset': 'IMPTC',
        'split': 'test',
        'checkpoint_selection': 'minimum validation NLL',
        'evaluation_seed': args.evaluation_seed,
        'device': str(device),
        'm1_run': str(m1_run),
        'm3_run': str(m3_run),
        'm1_best_epoch': int(m1_manifest['best_epoch']),
        'm3_best_epoch': int(m3_manifest['best_epoch']),
        'config_differences': differences,
        'results': results,
        'delta_m3_minus_m1': delta,
    }
    report = json.loads(json.dumps(report, default=json_safe))
    (output_dir / 'comparison.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    )
    (output_dir / 'config_difference.json').write_text(
        json.dumps(differences, indent=2, ensure_ascii=False) + '\n'
    )
    write_csv(output_dir / 'comparison.csv', results)
    plot_summary(output_dir / 'metric_comparison.png', results)
    print(json.dumps(results, indent=2))
    print(f'Comparison artifacts: {output_dir}')


if __name__ == '__main__':
    main()
