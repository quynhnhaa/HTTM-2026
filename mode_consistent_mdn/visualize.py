"""Plot actual run artifacts of the mode-consistent IMPTC experiment."""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PALETTE = ('#2563eb', '#d97706', '#16a34a')


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def load_prediction(run, name):
    path = run / 'fixed_samples/predictions' / f'{name}.npz'
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def plot_history(run, output):
    rows = read_csv(run / 'history.csv')
    if not rows:
        return
    epoch = np.array([int(row['epoch']) for row in rows])
    train = np.array([float(row['train_nll']) for row in rows])
    validation = np.array([float(row['validation_nll']) for row in rows])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epoch, train, label='Train', linewidth=1.8)
    ax.plot(epoch, validation, label='Validation', linewidth=1.8)
    ax.set(xlabel='Epoch', ylabel='Joint trajectory NLL / 48',
           title='NLL của biến thể theo epoch')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / '01_joint_nll.png', dpi=180)
    plt.close(fig)


def plot_metrics(run, output):
    files = sorted((run / 'metrics').glob('epoch_*.json'))
    if not files:
        return
    snapshots = [json.loads(path.read_text()) for path in files]
    epoch = [item['epoch'] for item in snapshots]
    metrics = [item['metrics'] for item in snapshots]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    groups = [
        [('minade20_m', 'minADE20'), ('minfde20_m', 'minFDE20')],
        [('ravg_percent', 'Ravg'), ('rmin_percent', 'Rmin')],
        [('s68_m2_per_s', 'S68'), ('s95_m2_per_s', 'S95')],
    ]
    units = ['m', '%', 'm²/s']
    titles = ['Sai số dịch chuyển', 'Reliability', 'Sharpness']
    for ax, group, unit, title in zip(axes, groups, units, titles):
        for field, label in group:
            values = [item.get(field, np.nan) for item in metrics]
            ax.plot(epoch, values, 'o-', label=label, markersize=3)
        ax.set(xlabel='Epoch', ylabel=unit, title=title)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Metric đánh giá của repo trên split eval tại các epoch được lưu')
    fig.tight_layout()
    fig.savefig(output / '02_repository_metrics.png', dpi=180)
    plt.close(fig)


def plot_mode_usage(run, output):
    files = sorted((run / 'fixed_samples/predictions').glob('epoch_*.npz'))
    if not files:
        return
    epochs = []
    mean_weights = []
    argmax_fractions = []
    for path in files:
        with np.load(path) as archive:
            weights = archive['pi'][:, 0, :]
            epochs.append(int(archive['epoch']))
            mean_weights.append(weights.mean(axis=0))
            argmax_fractions.append(np.bincount(
                weights.argmax(axis=-1), minlength=weights.shape[-1]
            ) / len(weights))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for mode, color in enumerate(PALETTE):
        axes[0].plot(epochs, np.array(mean_weights)[:, mode], 'o-',
                     color=color, label=f'Mode {mode+1}', markersize=3)
        axes[1].plot(epochs, np.array(argmax_fractions)[:, mode], 'o-',
                     color=color, label=f'Mode {mode+1}', markersize=3)
    axes[0].set_title('Trọng số mode trung bình')
    axes[1].set_title('Tỉ lệ là mode có trọng số cao nhất')
    for ax in axes:
        ax.set(xlabel='Epoch', ylim=(-.03, 1.03))
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel('Trên 8 mẫu validation cố định')
    fig.suptitle('Chẩn đoán sử dụng mode; 8 mẫu không đại diện toàn bộ IMPTC')
    fig.tight_layout()
    fig.savefig(output / '06_fixed_sample_mode_usage.png', dpi=180)
    plt.close(fig)


def plot_context(ax, observed, truth, prediction, sample_id):
    ax.plot(observed[:, 0], observed[:, 1], 'o-', color='#334155',
            markersize=2.5, linewidth=1.6, label='Quan sát')
    ax.plot(truth[:, 0], truth[:, 1], '-', color='#dc2626',
            linewidth=2, label='Tương lai thật')
    weights = prediction['pi'][0, 0]
    for mode, color in enumerate(PALETTE):
        mu = prediction['mu'][0, :, mode]
        ax.plot(mu[:, 0], mu[:, 1], '-', color=color, linewidth=1.5,
                alpha=.45 + .55 * weights[mode],
                label=f'Mode {mode + 1}')
    best = int(np.argmax(weights))
    mu = prediction['mu'][0, :, best]
    error = np.linalg.norm(mu - truth, axis=-1).mean()
    ax.plot(mu[:, 0], mu[:, 1], '--', color=PALETTE[best], linewidth=2.6)
    ax.set_title(f'{sample_id}\nADE tâm mode cao nhất: {error:.2f} m', fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.2)


def plot_fixed_overview(run, output, inputs, best):
    ids = inputs['sample_ids']
    n = len(ids)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= n:
            ax.axis('off')
            continue
        per_sample = {'mu': best['mu'][index:index+1],
                      'pi': best['pi'][index:index+1]}
        plot_context(ax, inputs['X'][index, :, :2], inputs['y'][index],
                     per_sample, str(ids[index]))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9)
    fig.suptitle('Cùng 8 mẫu validation đã chốt trước train — checkpoint best', fontsize=14)
    fig.tight_layout(rect=(0, .045, 1, .96))
    fig.savefig(output / '03_fixed_samples_best.png', dpi=180)
    plt.close(fig)


def plot_checkpoint_comparison(run, output, inputs, sample_index):
    candidates = ['epoch_0001', 'epoch_0005', 'epoch_0010', 'best', 'final']
    names = [name for name in candidates
             if (run / 'fixed_samples/predictions' / f'{name}.npz').exists()]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.6), squeeze=False)
    for name, ax in zip(names, axes.flat):
        snapshot = load_prediction(run, name)
        per_sample = {'mu': snapshot['mu'][sample_index:sample_index+1],
                      'pi': snapshot['pi'][sample_index:sample_index+1]}
        plot_context(ax, inputs['X'][sample_index, :, :2], inputs['y'][sample_index],
                     per_sample, name)
    fig.suptitle(f'Cùng mẫu {inputs["sample_ids"][sample_index]} qua các checkpoint', fontsize=14)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .92))
    fig.savefig(output / '04_same_sample_checkpoints.png', dpi=180)
    plt.close(fig)


def add_component_ellipse(ax, mean, covariance, color, weight):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0)
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    ellipse = Ellipse(
        mean, 2 * np.sqrt(eigenvalues[1]), 2 * np.sqrt(eigenvalues[0]),
        angle=angle, edgecolor=color, facecolor=color,
        linewidth=1.8, alpha=.15 + .35 * weight,
    )
    ax.add_patch(ellipse)


def plot_uncertainty(output, inputs, best, sample_index, horizon_indices):
    fig, axes = plt.subplots(1, len(horizon_indices), figsize=(5 * len(horizon_indices), 4.8))
    truth = inputs['y'][sample_index]
    observed = inputs['X'][sample_index, :, :2]
    for ax, h in zip(np.atleast_1d(axes), horizon_indices):
        ax.plot(observed[:, 0], observed[:, 1], '--', color='#334155', label='Quan sát')
        ax.plot(truth[:h+1, 0], truth[:h+1, 1], '-', color='#dc2626', label='GT')
        ax.scatter(truth[h, 0], truth[h, 1], color='#dc2626', marker='x', s=70)
        for k, color in enumerate(PALETTE):
            mean = best['mu'][sample_index, h, k]
            covariance = best['covariance'][sample_index, h, k]
            weight = float(best['pi'][sample_index, h, k])
            add_component_ellipse(ax, mean, covariance, color, weight)
            ax.scatter(*mean, color=color, s=35, label=f'Mode {k+1}: π={weight:.2f}')
        ax.set_title(f'h={h+1} ({(h+1)*.1:.1f} s)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Gaussian ellipse theo đầu ra thật: mỗi trục bán kính 1 độ lệch chuẩn\n'
                 'Ellipse của từng thành phần, không phải vùng tin cậy của toàn GMM')
    fig.tight_layout(rect=(0, 0, 1, .87))
    fig.savefig(output / '05_component_uncertainty.png', dpi=180)
    plt.close(fig)


def main(args):
    result_root = Path(os.environ.get(
        'MDN_RESULT_ROOT', PROJECT_ROOT / 'results/trained_models'
    )).expanduser().resolve()
    run = result_root / 'mode_consistent_mdn/imptc' / args.config_stem / 'runs' / args.run_id
    if not run.is_dir():
        raise FileNotFoundError(run)
    output = run / 'figures'
    output.mkdir(exist_ok=True)
    with np.load(run / 'fixed_samples/inputs.npz') as archive:
        inputs = {key: archive[key] for key in archive.files}
    if not 0 <= args.sample_index < len(inputs['sample_ids']):
        raise ValueError('sample_index outside fixed sample manifest')
    best = load_prediction(run, 'best')
    if not np.array_equal(inputs['sample_ids'], best['sample_ids']):
        raise ValueError('Fixed sample IDs differ between inputs and best prediction')
    plot_history(run, output)
    plot_metrics(run, output)
    plot_mode_usage(run, output)
    plot_fixed_overview(run, output, inputs, best)
    plot_checkpoint_comparison(run, output, inputs, args.sample_index)
    plot_uncertainty(output, inputs, best, args.sample_index, [7, 23, 47])
    print(f'Saved actual-output figures: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config-stem', default='mode_consistent_peds_imptc')
    parser.add_argument('--sample-index', type=int, default=0)
    main(parser.parse_args())
