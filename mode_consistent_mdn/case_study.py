"""Select and plot good, uncertain, and failure cases from the test split.

All ranking rules are explicit diagnostics, not additional official metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'base_mdn'))
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402
from model import ModeConsistentMDN  # noqa: E402


@torch.no_grad()
def diagnostic_scores(model, X, y, cfg, device):
    ade_values = []
    uncertainty_values = []
    batch_size = int(cfg.test_params['batch_size'])
    k = int(cfg.model_params['num_gaussians'])
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        inputs = torch.as_tensor(X[start:stop], dtype=torch.float32, device=device)
        target = torch.as_tensor(y[start:stop], dtype=torch.float32, device=device)
        params = decode_mdn_output(model(inputs), k, cfg.model_params.get('mdn_parameterization'))
        pi = params['pi'][:, 0, :]
        most_likely_mode = pi.argmax(dim=-1)
        mean = params['mu']
        choice = most_likely_mode[:, None, None, None].expand(-1, mean.shape[1], 1, 2)
        chosen_path = mean.gather(2, choice).squeeze(2)
        ade = torch.linalg.vector_norm(chosen_path - target, dim=-1).mean(dim=-1)
        # Trace of the final-step GMM covariance, including between-mode spread.
        final_mean = mean[:, -1, :, :]
        mixture_mean = (pi[..., None] * final_mean).sum(dim=1)
        within = torch.diagonal(params['covariance'][:, -1], dim1=-2, dim2=-1).sum(-1)
        between = (final_mean - mixture_mean[:, None, :]).square().sum(-1)
        uncertainty_trace = (pi * (within + between)).sum(-1)
        ade_values.append(ade.cpu().numpy())
        uncertainty_values.append(uncertainty_trace.cpu().numpy())
    return np.concatenate(ade_values), np.concatenate(uncertainty_values)


def select_cases(X, y, ade, uncertainty):
    movement = np.linalg.norm(y[:, -1] - X[:, -1, :2], axis=-1)
    eligible = np.flatnonzero(movement > 1.0)
    if len(eligible) < 3:
        raise ValueError('Fewer than three test tracks move over 1 metre')
    good = int(eligible[np.argmin(ade[eligible])])
    failure = int(eligible[np.argmax(ade[eligible])])
    lower, upper = np.quantile(ade[eligible], [.25, .75])
    middle = eligible[(ade[eligible] >= lower) & (ade[eligible] <= upper)]
    middle = middle[(middle != good) & (middle != failure)]
    uncertain = int(middle[np.argmax(uncertainty[middle])])
    return {'good': good, 'uncertain': uncertain, 'failure': failure}


def add_ellipse(ax, mean, covariance, color):
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    ax.add_patch(Ellipse(
        mean, 2 * np.sqrt(values[1]), 2 * np.sqrt(values[0]), angle=angle,
        edgecolor=color, facecolor=color, alpha=.17, linewidth=1.4,
    ))


@torch.no_grad()
def plot_cases(model, X, y, ids, selected, ade, uncertainty, output, device):
    colors = ['#2563eb', '#d97706', '#16a34a']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))
    for ax, (label, index) in zip(axes, selected.items()):
        inputs = torch.as_tensor(X[index:index+1], dtype=torch.float32, device=device)
        params = decode_mdn_output(model(inputs), 3, cfg.model_params.get('mdn_parameterization'))
        pi = params['pi'][0, 0].cpu().numpy()
        mu = params['mu'][0].cpu().numpy()
        cov = params['covariance'][0].cpu().numpy()
        observed = X[index, :, :2]
        truth = y[index]
        ax.plot(observed[:, 0], observed[:, 1], 'o-', color='#334155',
                linewidth=1.7, markersize=2.2, label='Quan sát')
        ax.plot(truth[:, 0], truth[:, 1], '-', color='#dc2626',
                linewidth=2, label='Tương lai thật')
        for mode, color in enumerate(colors):
            ax.plot(mu[:, mode, 0], mu[:, mode, 1], '-', color=color,
                    linewidth=1.8, label=f'Mode {mode+1}: π={pi[mode]:.2f}')
            add_ellipse(ax, mu[-1, mode], cov[-1, mode], color)
        ax.scatter(*truth[-1], color='#dc2626', marker='x', s=75)
        ax.set_title(f'{label.upper()} | {ids[index]}\n'
                     f'ADE tâm mode lớn nhất: {ade[index]:.2f} m; '
                     f'trace phương sai cuối: {uncertainty[index]:.2f} m²', fontsize=9)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Ba trường hợp chọn theo quy tắc cố định trên toàn bộ IMPTC test\n'
                 'Ellipse cuối: từng Gaussian, bán trục 1 độ lệch chuẩn; không phải vùng tin cậy GMM')
    fig.tight_layout(rect=(0, 0, 1, .86))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = PROJECT_ROOT / 'mode_consistent_mdn/configs/imptc/mode_consistent_peds_imptc.json'
    cfg = ConfigLoader(str(config_path), 'imptc', False, False,
                       config_path.stem, 'mode_consistent_mdn', 'testing')
    run = Path(cfg.result_path) / 'runs' / args.run_id
    checkpoint_path = run / 'checkpoints/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from utils.mdn_distribution import apply_checkpoint_parameterization
    apply_checkpoint_parameterization(cfg, checkpoint)
    if checkpoint.get('resolved_config', {}).get('experiment_params', {}).get('loss_name') != 'joint_trajectory_nll_divided_by_horizon':
        raise ValueError('Checkpoint does not belong to the mode-consistent method')
    model = ModeConsistentMDN(cfg.model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    loader = DataLoader(cfg)
    loader.load_test_data()
    X, y, _, _, _ = loader.get_test_data()
    ids = np.asarray(loader.sample_ids['test'])
    ade, uncertainty = diagnostic_scores(model, X, y, cfg, device)
    selected = select_cases(X, y, ade, uncertainty)
    report = {
        'schema_version': '1.0', 'split': 'test', 'checkpoint': str(checkpoint_path),
        'epoch': checkpoint['epoch'], 'selection': {
            'eligibility': 'ground-truth final displacement > 1 metre',
            'good': 'minimum ADE of highest-weight mode mean among eligible',
            'uncertain': 'maximum final GMM covariance trace within middle 50% ADE',
            'failure': 'maximum ADE of highest-weight mode mean among eligible',
        },
        'cases': [
            {'type': label, 'test_index': index, 'sample_id': str(ids[index]),
             'mode_mean_ade_m': float(ade[index]),
             'final_gmm_variance_trace_m2': float(uncertainty[index])}
            for label, index in selected.items()
        ],
        'diagnostic_only': True,
    }
    testing_dir = run / 'testing'
    figure_dir = run / 'figures'
    testing_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (testing_dir / 'case_selection.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    plot_cases(model, X, y, ids, selected, ade, uncertainty,
               figure_dir / '07_test_case_studies.png', device)
    print(f'Saved case study: {figure_dir / "07_test_case_studies.png"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu', default='0')
    main(parser.parse_args())
