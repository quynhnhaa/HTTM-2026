"""Plot training comparison, fixed IMPTC cases, and learned temporal attention."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'base_mdn'))
from model import AttentionMDN  # noqa: E402

BASELINE = ROOT / 'results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024'
ATTENTION = ROOT / 'results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs'


def history(run):
    with (run / 'history.csv').open(newline='') as handle:
        return list(csv.DictReader(handle))


def plot_losses(run, target):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for path, label, color in [(BASELINE, 'Baseline M3', '#2563eb'),
                               (run, 'LSTM–Attention–MDN', '#dc2626')]:
        rows = history(path)
        epochs = np.array([int(r['epoch']) for r in rows])
        for ax, field, title in zip(axes, ['train_nll', 'validation_nll'],
                                    ['Train NLL', 'Validation NLL']):
            ax.plot(epochs, [float(r[field]) for r in rows], label=label,
                    color=color, linewidth=1.2)
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('NLL / vị trí')
            ax.grid(alpha=.22)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(target, dpi=170)
    plt.close(fig)


def plot_metric_history(run, target):
    keys = [('minade20_m', 'minADE20 (m)'), ('minfde20_m', 'minFDE20 (m)'),
            ('ravg_percent', 'Ravg (%)'), ('s68_m2_per_s', 'S68 (m²/s)')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3))
    for path, label, color in [(BASELINE, 'Baseline M3', '#2563eb'),
                               (run, 'LSTM–Attention–MDN', '#dc2626')]:
        rows = [json.loads(p.read_text()) for p in sorted((path / 'metrics').glob('epoch_*.json'))]
        if not rows:
            continue
        for ax, (key, ylabel) in zip(axes.flat, keys):
            ax.plot([row['epoch'] for row in rows],
                    [row['metrics'][key] for row in rows], 'o-', color=color,
                    label=label, markersize=3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.22)
    axes[0, 0].legend()
    fig.suptitle('Metric trên validation ở các checkpoint định kỳ')
    fig.tight_layout()
    fig.savefig(target, dpi=170)
    plt.close(fig)


def mixture_mean(pred):
    return (pred['pi'][..., None] * pred['mu']).sum(axis=-2)


def plot_fixed_samples(run, target):
    a = np.load(run / 'fixed_samples/inputs.npz')
    b = np.load(BASELINE / 'fixed_samples/inputs.npz')
    if not all(np.array_equal(a[key], b[key]) for key in ('sample_ids', 'X', 'y')):
        raise ValueError('Fixed samples differ from baseline M3')
    pred_a = np.load(run / 'fixed_samples/predictions/best.npz')
    pred_b = np.load(BASELINE / 'fixed_samples/predictions/best.npz')
    mean_a, mean_b = mixture_mean(pred_a), mixture_mean(pred_b)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    for i, ax in enumerate(axes.flat):
        past, future = a['X'][i, :, :2], a['y'][i]
        ax.plot(past[:, 0], past[:, 1], '-', color='#475569', lw=1.6, label='Quan sát')
        ax.plot(future[:, 0], future[:, 1], '-', color='#16a34a', lw=1.8, label='Thực tế')
        ax.plot(mean_b[i, :, 0], mean_b[i, :, 1], '--', color='#2563eb',
                lw=1.4, label='Baseline M3')
        ax.plot(mean_a[i, :, 0], mean_a[i, :, 1], '-', color='#dc2626',
                lw=1.4, label='Attention MDN')
        ax.set_title(str(a['sample_ids'][i]), fontsize=8)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    fig.suptitle('Cùng 8 mẫu validation cố định; đường dự đoán là trung bình GMM từng bước')
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(target, dpi=170)
    plt.close(fig)


def plot_attention(run, target):
    cfg = json.loads((run / 'resolved_config.json').read_text())
    checkpoint = torch.load(run / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model = AttentionMDN(cfg['model_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    fixed = np.load(run / 'fixed_samples/inputs.npz')
    inputs = torch.as_tensor(fixed['X'], dtype=torch.float32)
    with torch.no_grad():
        states, _ = model.lstm(inputs)
        query = states[:, -1:, :] + model.future_queries.unsqueeze(0)
        _, weights = model.temporal_attention(query, states, states, need_weights=True)
    mean_weights = weights.mean(dim=0).numpy()
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(mean_weights, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('Bước quan sát (0–31)')
    ax.set_ylabel('Bước dự báo (0–47)')
    ax.set_title('Attention trung bình trên 8 mẫu validation cố định')
    fig.colorbar(im, ax=ax, label='Trọng số attention')
    fig.tight_layout()
    fig.savefig(target, dpi=170)
    plt.close(fig)


def main(args):
    global BASELINE
    run = Path(args.run_dir) if args.run_dir else ATTENTION / args.run_id
    if args.baseline_run:
        BASELINE = Path(args.baseline_run)
    from utils.mdn_distribution import resolve_parameterization
    policies = [resolve_parameterization(json.loads((p / 'resolved_config.json').read_text())['model_params'].get('mdn_parameterization')) for p in (run, BASELINE)]
    if policies[0] != policies[1]:
        raise ValueError('Visualization comparison needs runs with the same MDN parameterization')
    figures = run / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    plot_losses(run, figures / '01_train_validation_nll.png')
    plot_metric_history(run, figures / '02_official_metric_history.png')
    plot_fixed_samples(run, figures / '03_fixed_samples_baseline_vs_attention.png')
    plot_attention(run, figures / '04_temporal_attention.png')
    print(f'Saved four figures to {figures}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', help='Explicit attention run directory')
    parser.add_argument('--baseline-run', help='Explicit baseline run directory')
    parser.add_argument('--run-id', default='imptc_attention_seed2024')
    main(parser.parse_args())
