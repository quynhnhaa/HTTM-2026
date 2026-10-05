"""Compare baseline-selected fixed validation cases with K8 best checkpoint.
Run: python docs/sparsemax_mdn_k8/compare_fixed_cases.py
Only inference on eight saved inputs; no training or official test evaluation.
"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'base_mdn'))
from sparsemax_mdn.model import SparsemaxMDN
from utils.mdn_distribution import decode_mdn_output
BASE = ROOT / 'results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024'
SPARSE = ROOT / 'results/trained_models/sparsemax_mdn/imptc/sparsemax_k8_peds_imptc/runs/sparsemax_k8_v2_seed2024'
OUT = ROOT / 'docs/sparsemax_mdn_k8/figures'

def main():
    a = np.load(BASE / 'fixed_samples/inputs.npz')
    p = np.load(BASE / 'fixed_samples/predictions/best.npz')
    assert np.array_equal(a['sample_ids'], p['sample_ids'])
    checkpoint = SPARSE / 'checkpoints/best.pt'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['resolved_config']['model_params']
    assert cfg['num_gaussians'] == 8
    model = SparsemaxMDN(cfg).eval()
    model.load_state_dict(ck['model_state_dict'])
    with torch.inference_mode():
        q = {k: v.numpy() for k, v in decode_mdn_output(model(torch.as_tensor(a['X'], dtype=torch.float32)), 8, cfg.get('mdn_parameterization')).items()}
    means = [(p['pi'][..., None] * p['mu']).sum(2), (q['pi'][..., None] * q['mu']).sum(2)]
    errors = [np.linalg.norm(m - a['y'], axis=-1).mean(1) for m in means]
    order = np.argsort(errors[0])
    ids = [int(order[0]), int(order[len(order)//2]), int(order[-1])]
    active = (q['pi'] > 0).sum(-1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / '19_fixed_cases_k8_predictions.npz', sample_ids=a['sample_ids'], **q)
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.2))
    for col, (i, label) in enumerate(zip(ids, ['Sai số baseline thấp nhất', 'Sai số baseline ở giữa', 'Sai số baseline cao nhất'])):
        points = np.concatenate([a['X'][i, :, :2], a['y'][i], means[0][i], means[1][i]])
        lo, hi = points.min(0), points.max(0)
        center = (lo + hi) / 2
        # Identical limits across models; equal x/y scale prevents geometric distortion.
        half = max(float((hi-lo).max()) * .58, .08)
        for row in range(2):
            ax = axs[row, col]
            ax.plot(*a['X'][i, :, :2].T, color='#64748b', lw=2.2, label='Quan sát (3,2 s)')
            ax.plot(*a['y'][i].T, color='#159e77', lw=2.2, label='Thực tế (4,8 s)')
            ax.plot(*means[row][i].T, color=['#2563eb', '#ed713a'][row], ls='--', lw=2.2, label='Trung bình GMM')
            ax.scatter(*a['X'][i, -1, :2], color='black', s=22, zorder=5)
            ax.set_xlim(center[0]-half, center[0]+half)
            ax.set_ylim(center[1]-half, center[1]+half)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.18)
            ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
            name = 'Baseline M=3' if row == 0 else 'Sparsemax Kmax=8'
            extra = '' if row == 0 else f"\nK hoạt động TB={active[i].mean():.2f}"
            ax.set_title(f"{label}\n{name} | Mean-path ADE={errors[row][i]:.3f} m{extra}", fontsize=11)
    axs[0, 0].legend(fontsize=9, loc='best')
    fig.suptitle('So sánh trên cùng ba mẫu validation cố định', fontsize=20, weight='bold', y=.99)
    fig.text(.5, .015, 'Chọn theo sai số baseline trong 8 mẫu cố định; giữ nguyên mẫu và trục. Mean-path ADE khác minADE20 chính thức.', ha='center', fontsize=12)
    fig.tight_layout(rect=[0, .04, 1, .96])
    image = OUT / '19_fixed_cases_baseline_vs_sparsemax_k8.png'
    fig.savefig(image, dpi=180); plt.close(fig)
    metadata = {'script': str(Path(__file__).relative_to(ROOT)), 'image': str(image.relative_to(ROOT)), 'baseline_inputs': str((BASE/'fixed_samples/inputs.npz').relative_to(ROOT)), 'baseline_predictions': str((BASE/'fixed_samples/predictions/best.npz').relative_to(ROOT)), 'sparsemax_checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'selection': 'Same argsort and [0, len//2, -1] as baseline create_figures.py, among eight fixed validation samples', 'metric': 'Mean Euclidean error of mixture mean across all 48 future steps; not official minADE20', 'cases': [{'fixed_index': i, 'sample_id': str(a['sample_ids'][i]), 'baseline_mean_path_ade_m': float(errors[0][i]), 'sparsemax_mean_path_ade_m': float(errors[1][i]), 'sparsemax_active_k_mean': float(active[i].mean()), 'sparsemax_active_k_min': int(active[i].min()), 'sparsemax_active_k_max': int(active[i].max())} for i in ids]}
    image.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata['cases'], ensure_ascii=False, indent=2))
    print(image)

if __name__ == '__main__':
    main()
