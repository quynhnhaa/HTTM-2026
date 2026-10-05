"""Which component to split, and which K to keep. Validation data only."""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributions as dist

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from growing_mdn.model import num_components  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402

METRIC_KEYS = ('ravg_percent', 'rmin_percent', 's68_m2_per_s', 's95_m2_per_s')
MIN_MASS = 1e-8


@torch.no_grad()
def component_scores(model, X, y, device, batch_size):
    """Responsibility-weighted mean of -log N_k(y) per component.

    A high score means the component explains its assigned points poorly. The
    component index is a head index, not a physically consistent mode.
    """
    k = num_components(model)
    model.eval()
    mass = torch.zeros(k, dtype=torch.float64)
    weighted = torch.zeros(k, dtype=torch.float64)
    for start in range(0, len(X), batch_size):
        x = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        target = torch.as_tensor(y[start:start + batch_size], dtype=torch.float32, device=device)
        p = decode_mdn_output(model(x), k)
        log_n = dist.MultivariateNormal(p['mu'], p['covariance']).log_prob(target.unsqueeze(2))
        resp = torch.softmax(p['pi'].clamp_min(torch.finfo(p['pi'].dtype).tiny).log() + log_n, dim=-1)
        mass += resp.sum((0, 1)).double().cpu()
        weighted += (resp * -log_n).sum((0, 1)).double().cpu()
    scores = torch.full((k,), -math.inf, dtype=torch.float64)
    alive = mass > MIN_MASS
    scores[alive] = weighted[alive] / mass[alive]
    return scores.numpy()


@torch.no_grad()
def source_sigma_median(model, X, device, batch_size, source):
    """Median decoded sigma of component ``source`` over all samples: numpy [horizon, 2]."""
    k = num_components(model)
    model.eval()
    chunks = []
    for start in range(0, len(X), batch_size):
        x = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        chunks.append(decode_mdn_output(model(x), k)['sigma'][:, :, source].double().cpu())
    return torch.cat(chunks).median(dim=0).values.numpy()


def pick_split(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).any():
        raise ValueError('No component has a finite score; cannot choose a split')
    return int(np.argmax(scores))


def select_k(records, epsilon_nll, ravg_tolerance_pp, sharpness_tolerance_ratio):
    """Smallest K within tolerance of the best-NLL K on validation.

    ``records``: dicts with ``k``, ``validation_nll`` and ``METRIC_KEYS``.
    """
    if min(epsilon_nll, ravg_tolerance_pp, sharpness_tolerance_ratio) < 0:
        raise ValueError('Tolerances must be non-negative')
    for r in records:
        values = [r['validation_nll']] + [r[key] for key in METRIC_KEYS]
        if not all(np.isfinite(values)):
            raise ValueError(f'Non-finite validation values for K={r["k"]}')
    best = min(records, key=lambda r: r['validation_nll'])

    def acceptable(r):
        return (r['validation_nll'] <= best['validation_nll'] + epsilon_nll
                and r['ravg_percent'] >= best['ravg_percent'] - ravg_tolerance_pp
                and r['rmin_percent'] >= best['rmin_percent'] - ravg_tolerance_pp
                and r['s68_m2_per_s'] <= best['s68_m2_per_s'] + abs(best['s68_m2_per_s']) * sharpness_tolerance_ratio
                and r['s95_m2_per_s'] <= best['s95_m2_per_s'] + abs(best['s95_m2_per_s']) * sharpness_tolerance_ratio)

    candidates = sorted(r['k'] for r in records if acceptable(r))
    return {'selected_k': candidates[0], 'best_nll_k': best['k'], 'candidate_ks': candidates,
            'rule': {'epsilon_nll': epsilon_nll, 'ravg_tolerance_pp': ravg_tolerance_pp,
                     'sharpness_tolerance_ratio': sharpness_tolerance_ratio}}
