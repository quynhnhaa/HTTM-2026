"""Animate the attention MDN's fixed validation prediction over saved epochs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs'


def load_history(run):
    with (run / 'history.csv').open(newline='') as stream:
        return {int(row['epoch']): float(row['validation_nll'])
                for row in csv.DictReader(stream)}


def add_covariance_ellipse(ax, mean, covariance, color):
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 1e-12)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    patch = Ellipse(
        mean, 2 * np.sqrt(values[1]), 2 * np.sqrt(values[0]), angle=angle,
        edgecolor=color, facecolor=color, alpha=.10, linewidth=1.1,
    )
    ax.add_patch(patch)
    return patch


def output_files(run):
    folder = run / 'fixed_samples/predictions'
    found = {}
    for path in folder.glob('epoch_*.npz'):
        try:
            epoch = int(path.stem.split('_')[1])
        except (ValueError, IndexError):
            continue
        found[epoch] = path
    best_path = folder / 'best.npz'
    if best_path.exists():
        with np.load(best_path) as best:
            found[int(best['epoch'])] = best_path
    return sorted(found.items())


def main(args):
    run = Path(args.run_dir) if args.run_dir else RUNS / args.run_id
    fixed = np.load(run / 'fixed_samples/inputs.npz')
    if not 0 <= args.sample_index < len(fixed['X']):
        raise ValueError(f"sample-index must be 0..{len(fixed['X'])-1}")
    history = load_history(run)
    snapshots = output_files(run)
    if not snapshots:
        raise FileNotFoundError(f'No saved fixed-sample predictions in {run}')

    observed = fixed['X'][args.sample_index, :, :2]
    target = fixed['y'][args.sample_index]
    sample_id = str(fixed['sample_ids'][args.sample_index])
    colors = ['#2563eb', '#d97706', '#16a34a']
    frames = []
    all_xy = [observed, target]
    for epoch, path in snapshots:
        with np.load(path) as archive:
            data = {key: archive[key].copy() for key in
                    ('pi', 'mu', 'covariance')}
        item = {key: value[args.sample_index] for key, value in data.items()}
        item['epoch'] = epoch
        item['path'] = path
        item['expected'] = (item['pi'][..., None] * item['mu']).sum(axis=1)
        frames.append(item)
        all_xy.append(item['mu'].reshape(-1, 2))
        all_xy.append(item['expected'])

    joined = np.concatenate(all_xy)
    low, high = joined.min(axis=0), joined.max(axis=0)
    margin = np.maximum((high - low) * .08, .15)
    xlim = (low[0] - margin[0], high[0] + margin[0])
    ylim = (low[1] - margin[1], high[1] + margin[1])

    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    fig.subplots_adjust(bottom=.19, top=.86)
    ax.plot(observed[:, 0], observed[:, 1], 'o-', color='#334155',
            linewidth=1.8, markersize=2.5, label='Observed')
    ax.plot(target[:, 0], target[:, 1], '-', color='#111827',
            linewidth=2.1, label='Ground truth future')
    component_lines = [ax.plot([], [], '--', color=color, linewidth=1.2,
                               alpha=.72, label=f'Mode {i+1}') [0]
                       for i, color in enumerate(colors)]
    expected_line, = ax.plot([], [], '-', color='#dc2626', linewidth=2.4,
                             label='GMM expected path')
    endpoint_dots = [ax.plot([], [], 'o', color=color, markersize=4)[0]
                     for color in colors]
    title = ax.set_title('')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.22)
    ax.set_xlabel('x (ego coordinates, m)')
    ax.set_ylabel('y (ego coordinates, m)')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(labelsize=8)
    fig.text(.5, .045, 'Ellipse tại cuối quỹ đạo: 1 độ lệch chuẩn của từng Gaussian. '
             'Các frame là checkpoint đã lưu, không phải mọi epoch.',
             ha='center', fontsize=9)
    ellipse_artists = []

    def draw(frame_index):
        for artist in ellipse_artists:
            artist.remove()
        ellipse_artists.clear()
        frame = frames[frame_index]
        for mode, (line, dot, color) in enumerate(zip(component_lines, endpoint_dots, colors)):
            path = frame['mu'][:, mode, :]
            line.set_data(path[:, 0], path[:, 1])
            dot.set_data([path[-1, 0]], [path[-1, 1]])
            ellipse = add_covariance_ellipse(
                ax, path[-1], frame['covariance'][-1, mode], color
            )
            ellipse_artists.append(ellipse)
        expected = frame['expected']
        expected_line.set_data(expected[:, 0], expected[:, 1])
        epoch = frame['epoch']
        nll = history.get(epoch)
        nll_text = f' | validation NLL: {nll:.3f}' if nll is not None else ''
        pi_final = frame['pi'][-1]
        mode_text = ', '.join(f'π{i+1}={p:.2f}' for i, p in enumerate(pi_final))
        title.set_text(f'Epoch {epoch}{nll_text}\n{sample_id} | final-step {mode_text}')
        return component_lines + endpoint_dots + [expected_line, title] + ellipse_artists

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=900,
                              blit=False, repeat=True)
    out_dir = run / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f'05_learning_progress_sample{args.sample_index}.gif'
    animation.save(gif_path, writer=PillowWriter(fps=1.2))

    key_epochs = [1, 10, 100, 250, 500, 750, 1000, 1250, 1500, 1700, 1740]
    selected = [i for i, frame in enumerate(frames) if frame['epoch'] in key_epochs]
    if not selected:
        selected = list(range(min(9, len(frames))))
    columns = 3
    rows = (len(selected) + columns - 1) // columns
    fig2, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows), squeeze=False)
    for ax, i in zip(axes.flat, selected):
        frame = frames[i]
        ax.plot(observed[:, 0], observed[:, 1], '-', color='#334155', lw=1.2,
                label='Observed')
        ax.plot(target[:, 0], target[:, 1], '-', color='#111827', lw=1.5,
                label='Ground truth')
        for mode, color in enumerate(colors):
            ax.plot(frame['mu'][:, mode, 0], frame['mu'][:, mode, 1], '--',
                    color=color, lw=1.0, alpha=.72, label=f'Mode {mode+1}')
        pred = frame['expected']
        ax.plot(pred[:, 0], pred[:, 1], '-', color='#dc2626', lw=1.8,
                label='GMM expected path')
        nll = history.get(frame['epoch'])
        suffix = f' | val NLL {nll:.2f}' if nll is not None else ''
        ax.set_title(f"Epoch {frame['epoch']}{suffix}", fontsize=9)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.2)
    for ax in axes.flat[len(selected):]:
        ax.axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower center', ncol=5, fontsize=8)
    fig2.suptitle(f'Attention MDN: tiến trình dự báo trên cùng mẫu {sample_id}', y=.995)
    fig2.tight_layout(rect=(0, .035, 1, .975))
    png_path = out_dir / f'05_learning_progress_sample{args.sample_index}_montage.png'
    fig2.savefig(png_path, dpi=170)
    plt.close(fig2)
    plt.close(fig)
    print(f'Saved animation: {gif_path}')
    print(f'Saved montage: {png_path}')
    print('Saved epochs:', ', '.join(str(frame['epoch']) for frame in frames))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', help='Explicit run directory, including paper runs')
    parser.add_argument('--run-id', default='imptc_attention_seed2024')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Slot in the fixed validation sample manifest (0..7)')
    main(parser.parse_args())
