"""Exact masked mixture NLL: components with pi == 0 contribute nothing and get zero gradient."""
import sys
from pathlib import Path

import torch
import torch.distributions as dist

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.mdn_distribution import decode_mdn_output  # noqa: E402

TINY = 1e-30


def sparsemax_nll(raw, target, num_gaussians):
    """Returns (loss, diverged) like base_lstm.NLL_MDN_loss."""
    try:
        params = decode_mdn_output(raw, num_gaussians, {'mode': 'legacy'})
        components = dist.MultivariateNormal(params['mu'], params['covariance'])
        log_n = components.log_prob(target.unsqueeze(2))
    except (ValueError, RuntimeError):
        return None, True
    pi = params['pi']
    # clamp before log so the masked branch never produces inf * 0 = NaN in backward
    log_pi = torch.where(pi > 0, torch.log(pi.clamp_min(TINY)), torch.full_like(pi, float('-inf')))
    # Masked entries: -inf + finite = -inf; their logsumexp weight is exactly 0.
    loss = -torch.logsumexp(log_pi + log_n, dim=-1).mean()
    if not torch.isfinite(loss):
        return None, True
    return loss, False
