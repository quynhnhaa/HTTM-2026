#!/usr/bin/env python3
"""Train, evaluate, and summarize the ETH/UCY Gaussian-count ablation.

The experiment preserves each official repository fold configuration and only
changes ``model_params.num_gaussians``. Test metrics are produced by the
repository evaluator from the validation-best checkpoint.
"""

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
from train import training
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.experiment import set_global_seed
from utils.helper import count_model_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / 'stable_mdn' / 'configs' / 'eth'
DEFAULT_OUTPUT = PROJECT_ROOT / 'results' / 'comparisons' / 'eth_ucy_m_ablation'
FOLDS = ('eth', 'hotel', 'univ', 'zara1', 'zara2')
MIXTURES = (1, 2, 3, 5, 8)
PAPER_METHODS = {
    'Trajectron++': {'ravg_percent': 64.9, 'rmin_percent': 30.9, 'minade20_m': 0.30, 'minfde20_m': 0.51},
    'Social-Implicit': {'ravg_percent': 82.7, 'rmin_percent': 62.5, 'minade20_m': 0.33, 'minfde20_m': 0.67},
    'MID': {'ravg_percent': 84.6, 'rmin_percent': 65.9, 'minade20_m': 0.21, 'minfde20_m': 0.38},
    'FlowChain': {'ravg_percent': 86.5, 'rmin_percent': 62.2, 'minade20_m': 0.29, 'minfde20_m': 0.52},
    'Paper Ours (M=3)': {'ravg_percent': 91.1, 'rmin_percent': 72.5, 'minade20_m': 0.26, 'minfde20_m': 0.50},
}


def parse_csv_ints(value):
    return tuple(int(part.strip()) for part in value.split(',') if part.strip())


def parse_csv_strings(value):
    return tuple(part.strip().lower() for part in value.split(',') if part.strip())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'status', 'train', 'evaluate', 'summarize', 'all'))
    parser.add_argument('--folds', default=','.join(FOLDS))
    parser.add_argument('--mixtures', default=','.join(map(str, MIXTURES)))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--force', action='store_true', help='Re-run completed evaluation jobs; training artifacts are never overwritten.')
    return parser.parse_args()


def validate_selection(folds, mixtures):
    invalid_folds = sorted(set(folds) - set(FOLDS))
    invalid_m = sorted(set(mixtures) - set(MIXTURES))
    if invalid_folds or invalid_m:
        raise ValueError(f'Invalid folds={invalid_folds}, mixtures={invalid_m}')


def config_name(fold, mixtures):
    return f'eth_ucy_{fold}_m{mixtures}'


def run_id(fold, mixtures, seed):
    return f'eth_ucy_{fold}_m{mixtures}_seed{seed}'


def build_config(fold, mixtures, seed, kind='training'):
    path = CONFIG_DIR / f'default_peds_eth_{fold}.json'
    cfg = ConfigLoader(
        config_path=str(path), target='eth', with_log=True, with_print=True,
        name=config_name(fold, mixtures), model_arch='stable_mdn', type=kind,
    )
    cfg.model_params['num_gaussians'] = int(mixtures)
    cfg.experiment_params = {
        'experiment_name': 'eth_ucy_mixture_ablation',
        'seed': int(seed),
        'fixed_sample_count': 8,
        'fixed_sample_id_prefix': f'eth_ucy_{fold}',
        'checkpoint_epochs': [1, 5, 10, 100, 300, 500, 1000, 1500, 2000, 2500],
        'checkpoint_every': 0,
        'metric_every': 0,
        'save_last_every': 1,
    }
    return cfg


def get_run_dir(cfg, fold, mixtures, seed):
    return Path(cfg.result_path) / 'runs' / run_id(fold, mixtures, seed)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare(output_dir, folds, mixtures, seed):
    jobs = []
    for fold in folds:
        for m in mixtures:
            cfg = build_config(fold, m, seed)
            jobs.append({
                'fold': fold, 'num_gaussians': m, 'seed': seed,
                'base_config': str(CONFIG_DIR / f'default_peds_eth_{fold}.json'),
                'run_id': run_id(fold, m, seed),
                'run_dir': str(get_run_dir(cfg, fold, m, seed)),
                'train_samples': cfg.paths['train_data_paths'],
                'validation_samples': cfg.paths['val_data_paths'],
                'test_samples': cfg.paths['test_data_paths'],
            })
    protocol = {
        'schema_version': '1.0',
        'experiment': 'ETH/UCY Gaussian-count ablation',
        'selection_rule': 'Select M by the lowest mean validation NLL across five LOO folds; test metrics are reported after selection.',
        'constant_parameters': 'All per-fold repository settings are preserved; only model_params.num_gaussians changes within a fold.',
        'checkpoint_selection': 'minimum validation NLL',
        'test_metrics': ['Ravg', 'Rmin', 'S68', 'S95', 'minADE20', 'minFDE20'],
        'test_horizons_seconds': [0.8, 1.6, 2.4, 3.2, 4.0, 4.8],
        'jobs': jobs,
    }
    write_json(output_dir / 'experiment_matrix.json', protocol)
    return protocol


def job_status(cfg, fold, m, seed, output_dir):
    directory = get_run_dir(cfg, fold, m, seed)
    manifest_path = directory / 'run_manifest.json'
    evaluation_path = output_dir / 'evaluations' / f'm{m}' / fold / 'metrics.json'
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    return {
        'fold': fold, 'num_gaussians': m,
        'train_status': manifest.get('status', 'not_started'),
        'last_epoch': manifest.get('final_epoch', manifest.get('last_completed_epoch')),
        'best_epoch': manifest.get('best_epoch'),
        'best_validation_nll': manifest.get('best_validation_nll'),
        'evaluated': evaluation_path.is_file(),
        'run_dir': str(directory),
    }


def train_job(cfg, fold, m, seed, force=False):
    directory = get_run_dir(cfg, fold, m, seed)
    manifest_path = directory / 'run_manifest.json'
    if manifest_path.is_file():
        if force:
            raise ValueError(
                f'Refusing to overwrite training artifacts in {directory}. '
                'Use a different --seed or MDN_RESULT_ROOT for a fresh run.'
            )
        status = read_json(manifest_path).get('status')
        if status in ('completed', 'diverged'):
            print(f'[skip] terminal train ({status}): fold={fold}, M={m}')
            return
    previous_run = os.environ.get('MDN_RUN_ID')
    previous_resume = os.environ.get('MDN_RESUME_CHECKPOINT')
    os.environ['MDN_RUN_ID'] = run_id(fold, m, seed)
    last = directory / 'checkpoints' / 'last.pt'
    if last.is_file() and not force:
        os.environ['MDN_RESUME_CHECKPOINT'] = str(last)
        print(f'[resume] fold={fold}, M={m}, checkpoint={last}')
    else:
        os.environ.pop('MDN_RESUME_CHECKPOINT', None)
    try:
        training(cfg=cfg, gpu_id=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    finally:
        if previous_run is None:
            os.environ.pop('MDN_RUN_ID', None)
        else:
            os.environ['MDN_RUN_ID'] = previous_run
        if previous_resume is None:
            os.environ.pop('MDN_RESUME_CHECKPOINT', None)
        else:
            os.environ['MDN_RESUME_CHECKPOINT'] = previous_resume


def full_test_nll(model, loader, cfg, device):
    data_x, data_y, _, _, _ = loader.get_test_data()
    total, values = 0.0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data_x), cfg.test_params['batch_size']):
            x = torch.as_tensor(data_x[start:start + cfg.test_params['batch_size']], dtype=torch.float32, device=device)
            y = torch.as_tensor(data_y[start:start + cfg.test_params['batch_size']], dtype=torch.float32, device=device)
            loss, diverged = NLL_MDN_loss(model(x), y, cfg.model_params['num_gaussians'], cfg.model_params.get('mdn_parameterization'))
            if diverged or loss is None or not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite test NLL for {cfg.name}')
            count = y.shape[0] * y.shape[1]
            total += float(loss) * count
            values += count
    return total / values


def evaluate_job(cfg, fold, m, seed, output_dir, device, force=False):
    destination = output_dir / 'evaluations' / f'm{m}' / fold
    metrics_path = destination / 'metrics.json'
    if metrics_path.is_file() and not force:
        print(f'[skip] completed evaluation: fold={fold}, M={m}')
        return read_json(metrics_path)
    directory = get_run_dir(cfg, fold, m, seed)
    checkpoint_path = directory / 'checkpoints' / 'best.pt'
    manifest_path = directory / 'run_manifest.json'
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f'Missing completed run artifacts: {directory}')
    manifest = read_json(manifest_path)
    if manifest.get('status') not in ('completed', 'diverged'):
        raise ValueError(f'Run has neither completed nor terminated with divergence: {directory}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint['resolved_config']['model_params'] != cfg.model_params:
        raise ValueError(f'Checkpoint/config model mismatch: {checkpoint_path}')

    destination.mkdir(parents=True, exist_ok=True)
    cfg.testing_path = str(destination)
    cfg.test_ego_examples_path = str(destination / 'examples' / 'ego')
    cfg.test_world_examples_path = str(destination / 'examples' / 'world')
    Path(cfg.test_ego_examples_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.test_world_examples_path).mkdir(parents=True, exist_ok=True)
    loader = DataLoader(cfg=cfg)
    loader.load_test_data()
    model = LSTM_Trajectory_Forecast(cfg=cfg.model_params).to(device)
    model.load_state_dict(checkpoint['model'])
    test_nll = full_test_nll(model, loader, cfg, device)
    set_global_seed(seed)
    official = MDN_Forecaster(
        cfg=cfg, model=model, data_loader=loader, type='testing', device=device,
        logger=logging.getLogger(f'eth_ucy_{fold}_m{m}'),
    ).evaluate(epoch=None)
    result = {
        'schema_version': '1.0', 'dataset': 'ETH/UCY', 'split': 'test',
        'fold': fold, 'num_gaussians': m, 'seed': seed,
        'checkpoint_selection': 'minimum validation NLL',
        'training_status': manifest['status'],
        'checkpoint': str(checkpoint_path), 'checkpoint_epoch': int(checkpoint['epoch']),
        'validation_nll': float(checkpoint['best_validation_nll']),
        'test_nll': float(test_nll),
        'parameter_count': int(count_model_parameters(model)),
        **official,
    }
    write_json(metrics_path, result)
    return result


def write_csv(path, rows):
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def summarize(output_dir, folds, mixtures):
    rows = []
    for m in mixtures:
        for fold in folds:
            path = output_dir / 'evaluations' / f'm{m}' / fold / 'metrics.json'
            if not path.is_file():
                raise FileNotFoundError(f'Missing evaluation: {path}')
            rows.append(read_json(path))
    scalar_metrics = (
        'validation_nll', 'test_nll', 'ravg_percent', 'rmin_percent',
        's68_m2_per_s', 's95_m2_per_s', 'minade20_m', 'minfde20_m',
        'parameter_count', 'inference_ms_per_batch',
    )
    averages = []
    for m in mixtures:
        selected = [row for row in rows if row['num_gaussians'] == m]
        average = {'num_gaussians': m, 'fold_count': len(selected)}
        for metric in scalar_metrics:
            average[metric] = float(np.mean([row[metric] for row in selected]))
        averages.append(average)
    selected_m = min(averages, key=lambda row: row['validation_nll'])['num_gaussians']
    report = {
        'schema_version': '1.0', 'dataset': 'ETH/UCY', 'aggregation': 'unweighted arithmetic mean across LOO folds',
        'selection_metric': 'mean validation NLL', 'selected_num_gaussians': selected_m,
        'fold_results': rows, 'mean_results': averages, 'paper_table_6_average': PAPER_METHODS,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'summary.json', report)
    write_csv(output_dir / 'fold_results.csv', rows)
    write_csv(output_dir / 'mean_results.csv', averages)
    plot_metric_curves(output_dir / 'metrics_vs_m.png', averages, selected_m)
    plot_tradeoff(output_dir / 'accuracy_reliability_tradeoff.png', averages, selected_m)
    plot_paper_comparison(output_dir / 'paper_comparison.png', averages, selected_m)
    print(f'Selected M={selected_m} by lowest mean validation NLL.')
    return report


def plot_metric_curves(path, rows, selected_m):
    panels = [
        ('validation_nll', 'Validation NLL ↓'), ('minade20_m', 'minADE20 [m] ↓'),
        ('minfde20_m', 'minFDE20 [m] ↓'), ('ravg_percent', 'Ravg [%] ↑'),
        ('rmin_percent', 'Rmin [%] ↑'), ('s68_m2_per_s', 'S68 [m²/s] ↓*'),
    ]
    x = [row['num_gaussians'] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (metric, title) in zip(axes.flat, panels):
        y = [row[metric] for row in rows]
        ax.plot(x, y, marker='o', linewidth=2)
        chosen = x.index(selected_m)
        ax.scatter([x[chosen]], [y[chosen]], color='#D55E00', s=80, zorder=3, label=f'Selected M={selected_m}')
        ax.set_title(title); ax.set_xlabel('Number of Gaussians M'); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle('ETH/UCY mixture-count ablation — mean over 5 LOO folds\n*S68 is interpreted only together with reliability')
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches='tight'); plt.close(fig)


def plot_tradeoff(path, rows, selected_m):
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        m = row['num_gaussians']
        ax.scatter(row['minade20_m'], row['ravg_percent'], s=70 + 10*m,
                   color='#D55E00' if m == selected_m else '#0072B2')
        ax.annotate(f'M={m}', (row['minade20_m'], row['ravg_percent']), xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('minADE20 [m] — lower is better'); ax.set_ylabel('Ravg [%] — higher is better')
    ax.set_title('Accuracy–calibration trade-off on ETH/UCY'); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches='tight'); plt.close(fig)


def plot_paper_comparison(path, rows, selected_m):
    by_m = {row['num_gaussians']: row for row in rows}
    methods = dict(PAPER_METHODS)
    if 3 in by_m:
        methods['Reproduction (M=3)'] = by_m[3]
    methods[f'Selected (M={selected_m})'] = by_m[selected_m]
    metrics = [('ravg_percent', 'Ravg [%] ↑'), ('rmin_percent', 'Rmin [%] ↑'),
               ('minade20_m', 'minADE20 [m] ↓'), ('minfde20_m', 'minFDE20 [m] ↓')]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    labels = list(methods)
    colors = ['#999999'] * len(PAPER_METHODS) + ['#0072B2'] * (len(methods) - len(PAPER_METHODS))
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = [methods[label][metric] for label in labels]
        bars = ax.barh(labels, values, color=colors)
        ax.bar_label(bars, fmt='%.3g', padding=3, fontsize=8); ax.set_title(title); ax.grid(axis='x', alpha=0.25)
    fig.suptitle('Paper Table 6 averages vs our ETH/UCY reproduction')
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches='tight'); plt.close(fig)


def main():
    args = parse_args()
    folds = parse_csv_strings(args.folds)
    mixtures = parse_csv_ints(args.mixtures)
    validate_selection(folds, mixtures)
    output_dir = args.output_dir.expanduser().resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prepare(output_dir, folds, mixtures, args.seed)

    if args.action == 'prepare':
        print(f'Experiment matrix written to {output_dir / "experiment_matrix.json"}')
        return
    if args.action == 'status':
        statuses = [job_status(build_config(fold, m, args.seed), fold, m, args.seed, output_dir) for fold in folds for m in mixtures]
        write_json(output_dir / 'status.json', statuses)
        for row in statuses:
            print(f"{row['fold']:6s} M={row['num_gaussians']}: train={row['train_status']}, evaluated={row['evaluated']}")
        return
    if args.action in ('train', 'all'):
        for m in mixtures:
            for fold in folds:
                train_job(build_config(fold, m, args.seed), fold, m, args.seed, args.force)
    if args.action in ('evaluate', 'all'):
        for m in mixtures:
            for fold in folds:
                evaluate_job(build_config(fold, m, args.seed, 'testing'), fold, m, args.seed, output_dir, device, args.force)
    if args.action in ('summarize', 'all'):
        summarize(output_dir, folds, mixtures)


if __name__ == '__main__':
    main()
