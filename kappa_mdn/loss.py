"""Exact mixture NLL plus the price of the soft number of open components (training only)."""
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


def kappa_nll(raw, target, num_gaussians, unnormalised_log_weights=False):
    """Mean negative log-likelihood of the mixture; components with pi == 0 contribute exactly nothing.

    unnormalised_log_weights=True (training): the last K raw values are log-weights used as they are, without a
    softmax, so a closed gate removes probability mass and costs NLL (the weights then sum to less than one).
    """
    try:
        params = decode_mdn_output(raw, num_gaussians, {'mode': 'legacy'})
        log_n = dist.MultivariateNormal(params['mu'], params['covariance']).log_prob(target.unsqueeze(2))
    except (ValueError, RuntimeError):
        return None, True
    pi = params['pi']
    if unnormalised_log_weights:
        log_pi = raw[..., 5 * num_gaussians:]
    else:
        log_pi = torch.where(pi > 0, torch.log(pi.clamp_min(TINY)), torch.full_like(pi, float('-inf')))
    loss = -torch.logsumexp(log_pi + log_n, dim=-1).mean()
    if not torch.isfinite(loss):
        return None, True
    return loss, False


class LossStats:
    """Per-batch pure NLL and penalty of the current training epoch (reset when consumed)."""

    def __init__(self):
        self.nll, self.penalty = [], []

    def record(self, nll, penalty):
        self.nll.append(float(nll))
        self.penalty.append(float(penalty))

    def consume(self):
        if not self.nll:
            return None
        out = {'train_nll': sum(self.nll) / len(self.nll), 'train_penalty': sum(self.penalty) / len(self.penalty)}
        self.nll, self.penalty = [], []
        return out


def make_loss_fn(model, num_gaussians, penalty_lambda, stats):
    """loss_fn(output, target) -> (loss, diverged) as MDN_Trainer expects.

    In training mode the objective is NLL + lambda * soft_open_count; in eval mode (validation) only the NLL.
    """
    def loss_fn(output, target):
        nll, diverged = kappa_nll(output, target, num_gaussians, unnormalised_log_weights=model.training)
        if diverged:
            return None, True
        if not model.training:
            return nll, False
        penalty = penalty_lambda * model.soft_open_count()
        stats.record(nll.detach(), penalty.detach())
        return nll + penalty, False
    return loss_fn
