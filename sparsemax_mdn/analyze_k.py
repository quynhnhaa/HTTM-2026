"""Exploratory, read-only analysis of K(x, t) = #{k : pi_k > 0} on the VALIDATION split only."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def rankdata(a):
    """Average ranks (ties share the mean rank), numpy only."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a))
    sorted_a = a[order]
    start = 0
    for i in range(1, len(a) + 1):
        if i == len(a) or sorted_a[i] != sorted_a[start]:
            ranks[order[start:i]] = (start + i - 1) / 2.0 + 1.0
            start = i
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None  # undefined for a constant variable
    return float(np.corrcoef(rx, ry)[0, 1])


def per_sample_statistics(pi, mu, covariance, y):
    """pi [N,T,K], mu [N,T,K,2], covariance [N,T,K,2,2], y [N,T,2]."""
    support = (pi > 0).sum(-1)  # [N, T]
    top = pi.argmax(-1)  # [N, T]
    n, t = top.shape
    top_mu = np.take_along_axis(mu, top[..., None, None], axis=2)[:, :, 0]  # [N, T, 2]
    ade_top = np.linalg.norm(top_mu - y, axis=-1).mean(-1)
    trace = covariance[..., 0, 0] + covariance[..., 1, 1]
    spread = (pi * np.sqrt(trace)).sum(-1).mean(-1)
    return support, ade_top, spread


def analyze(pi, mu, covariance, y, k_max):
    support, ade_top, spread = per_sample_statistics(pi, mu, covariance, y)
    flat = support.ravel()
    k_sample = support.mean(-1)
    active = (pi > 0).reshape(-1, k_max).mean(0)
    return {
        'status': 'exploratory; validation split only; K is a head-index count, not a physical mode count',
        'num_validation_samples': int(len(pi)), 'k_max': int(k_max),
        'k_distribution': {
            'histogram': {str(k): int((flat == k).sum()) for k in range(0, k_max + 1)},
            'fraction': {str(k): float((flat == k).mean()) for k in range(0, k_max + 1)},
            'mean': float(flat.mean()), 'std': float(flat.std())},
        'mean_k_per_forecast_step': support.mean(0).tolist(),
        'component_activity': {
            'fraction_active_per_component': active.tolist(),
            'dead_components': int((active == 0).sum())},
        'association_with_difficulty': {
            'method': 'Spearman correlation (numpy ranks) between per-sample mean K and the proxy; null if undefined',
            'spearman_k_vs_top_component_ade': spearman(k_sample, ade_top),
            'spearman_k_vs_weighted_sqrt_trace': spearman(k_sample, spread),
            'proxy_top_component_ade': 'ADE between ground truth and mean of the highest-weight component over the steps',
            'proxy_weighted_sqrt_trace': 'mean over steps of sum_k pi_k * sqrt(trace(Sigma_k))'}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--checkpoint', default='best')
    parser.add_argument('--gpu', default='-1')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--config')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import torch
    from utils.config_loader import ConfigLoader
    from utils.data_loader import DataLoader
    from utils.mdn_distribution import decode_mdn_output
    from sparsemax_mdn.artifacts import ARCH
    from sparsemax_mdn.model import SparsemaxMDN

    result_root = ROOT / 'results/trained_models/sparsemax_mdn/imptc'
    matches = [p for p in result_root.glob(f'*/runs/{args.run_id}')
               if args.config is None or p.parents[1].name == args.config]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected exactly one run {args.run_id!r}, found {matches}')
    run = matches[0]
    config_name = run.parents[1].name
    cfg = ConfigLoader(str(ROOT / 'sparsemax_mdn/configs/imptc' / f'{config_name}.json'), 'imptc',
                       False, False, config_name, 'sparsemax_mdn', 'training')
    saved = torch.load(run / 'checkpoints' / f'{args.checkpoint}.pt', map_location='cpu', weights_only=False)
    if saved.get('architecture') != ARCH:
        raise ValueError(f"Unexpected architecture {saved.get('architecture')!r}")
    k_max = cfg.model_params['num_gaussians']
    model = SparsemaxMDN(cfg.model_params)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()

    loader = DataLoader(cfg)
    loader.load_eval_data()  # validation split only
    X, y = loader.eval_data[0], loader.eval_data[1]
    if args.limit:
        X, y = X[:args.limit], y[:args.limit]
    parts = {key: [] for key in ('pi', 'mu', 'covariance')}
    with torch.no_grad():
        for start in range(0, len(X), 512):
            raw = model(torch.as_tensor(X[start:start + 512], dtype=torch.float32))
            decoded = decode_mdn_output(raw, k_max)
            for key in parts:
                parts[key].append(decoded[key].numpy())
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    result = analyze(arrays['pi'].astype(np.float64), arrays['mu'].astype(np.float64),
                     arrays['covariance'].astype(np.float64), np.asarray(y, dtype=np.float64), k_max)
    result.update(run_id=args.run_id, checkpoint=args.checkpoint, checkpoint_epoch=int(saved['epoch']),
                  split='validation', limit=args.limit)
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'k_usage.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'wrote {out / "k_usage.json"}')


if __name__ == '__main__':
    main()
