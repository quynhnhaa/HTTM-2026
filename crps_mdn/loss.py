"""NLL + lambda * (CRPS_x + CRPS_y) for the baseline bivariate Gaussian-mixture MDN.

The marginal of a bivariate Gaussian mixture on one axis is a 1-D Gaussian mixture with the same weights, means
(mu_x or mu_y) and standard deviations (sigma_x or sigma_y); rho does not enter. The CRPS of a 1-D Gaussian mixture has
the closed form of Grimit et al. (2006):

    CRPS(F, y) = sum_i pi_i A(y - mu_i, s_i^2) - 1/2 sum_ij pi_i pi_j A(mu_i - mu_j, s_i^2 + s_j^2)
    A(m, s^2)  = m (2 Phi(m / s) - 1) + 2 s phi(m / s)
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from base_lstm import NLL_MDN_loss  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _abs_mean(m, s):
    """A(m, s^2) = E|N(m, s^2)|, i.e. m (2 Phi(m/s) - 1) + 2 s phi(m/s); s > 0, broadcasting shapes."""
    z = m / s
    return m * torch.erf(z / SQRT2) + 2.0 * s * INV_SQRT_2PI * torch.exp(-0.5 * z * z)


def mixture_crps_1d(pi, mu, sigma, y):
    """Exact CRPS of a 1-D Gaussian mixture. pi, mu, sigma: [..., K]; y: [...]; returns [...]."""
    first = (pi * _abs_mean(y.unsqueeze(-1) - mu, sigma)).sum(-1)
    diff = mu.unsqueeze(-1) - mu.unsqueeze(-2)                                   # [..., K, K]
    scale = torch.sqrt(sigma.unsqueeze(-1).square() + sigma.unsqueeze(-2).square())
    second = (pi.unsqueeze(-1) * pi.unsqueeze(-2) * _abs_mean(diff, scale)).sum((-1, -2))
    return first - 0.5 * second


def crps_xy(output, target, num_gaussians):
    """Mean over samples and forecast steps of CRPS_x + CRPS_y of the decoded legacy mixture."""
    params = decode_mdn_output(output, num_gaussians, {'mode': 'legacy'})
    pi, mu, sigma = params['pi'], params['mu'], params['sigma']                  # [B,T,K], [B,T,K,2], [B,T,K,2]
    crps_x = mixture_crps_1d(pi, mu[..., 0], sigma[..., 0], target[..., 0])
    crps_y = mixture_crps_1d(pi, mu[..., 1], sigma[..., 1], target[..., 1])
    return (crps_x + crps_y).mean()


class LossStats:
    """Per-batch pure NLL and CRPS of the current training epoch (reset when consumed)."""

    def __init__(self):
        self.nll, self.crps = [], []

    def record(self, nll, crps):
        self.nll.append(float(nll))
        self.crps.append(float(crps))

    def consume(self):
        if not self.nll:
            return None
        out = {'train_nll': sum(self.nll) / len(self.nll), 'train_crps': sum(self.crps) / len(self.crps)}
        self.nll, self.crps = [], []
        return out


def make_loss_fn(model, num_gaussians, crps_lambda, stats, parameterization=None):
    """loss_fn(output, target) -> (loss, diverged) as MDN_Trainer expects.

    Training: NLL + lambda * CRPS. Evaluation mode (validation): the pure NLL, exactly the baseline loss.
    """
    def loss_fn(output, target):
        nll, diverged = NLL_MDN_loss(output, target, num_gaussians, parameterization)
        if diverged:
            return None, True
        if not model.training:
            return nll, False
        crps = crps_xy(output, target, num_gaussians)
        if not torch.isfinite(crps):
            return None, True
        stats.record(nll.detach(), crps.detach())
        return nll + crps_lambda * crps, False
    return loss_fn
