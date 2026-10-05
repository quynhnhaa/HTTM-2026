"""LSTM-MDN whose number of components is a learnable real parameter kappa (one value per forecast step).

kappa = 1 + (K_max - 1) * sigmoid(kappa_logit), so it always stays inside [1, K_max].
Training uses soft gates, gate[t, j] = sigmoid((kappa[t] - j + 0.5) / tau), j = 1..K_max, multiplied onto the
softmax weights WITHOUT renormalising (closing a component that carries mass loses mass, which the NLL penalises);
evaluation uses hard gates, K(t) = round(kappa[t]) with exact-zero weights above K(t) and softmax over the open ones.
The raw output keeps the baseline layout [B, H, 6K] so every baseline consumer works unchanged.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402

SENTINEL = -1e9  # finite stand-in for log(0); exp(SENTINEL - max) == 0 exactly in the baseline softmax


def component_index(k_max, like):
    return torch.arange(1, k_max + 1, device=like.device, dtype=like.dtype)


def log_gate(kappa, tau, k_max):
    """log gate[t, j] = logsigmoid((kappa[t] - j + 0.5) / tau); shape [T, K]."""
    j = component_index(k_max, kappa)
    return F.logsigmoid((kappa[:, None] - j[None, :] + 0.5) / tau)


def hard_k(kappa, k_max):
    """Number of open components per forecast step (integer-valued float), at least 1."""
    return torch.round(kappa).clamp(1, k_max)


def hard_open(kappa, k_max):
    j = component_index(k_max, kappa)
    return j[None, :] <= hard_k(kappa, k_max)[:, None]


class KappaMDN(LSTM_Trajectory_Forecast):
    def __init__(self, cfg):
        if cfg.get('mdn_parameterization', {'mode': 'legacy'}) != {'mode': 'legacy'}:
            raise ValueError('KappaMDN requires the legacy MDN parameterization')
        super().__init__(cfg)
        self.k_max = int(cfg['num_gaussians'])
        settings = cfg.get('kappa', {})
        init = float(settings.get('init', self.k_max / 2))
        if not 1.0 < init < self.k_max:
            raise ValueError(f'kappa init must lie strictly inside (1, K_max), got {init}')
        fraction = torch.tensor((init - 1.0) / (self.k_max - 1.0))
        self.kappa_logit = nn.Parameter(torch.full((int(cfg['forecast_horizon']),), float(torch.logit(fraction))))
        self.register_buffer('tau', torch.tensor(float(settings.get('tau_start', 1.0))))

    @property
    def kappa(self):
        """Real-valued number of components per forecast step, inside [1, K_max]."""
        return 1.0 + (self.k_max - 1.0) * torch.sigmoid(self.kappa_logit)

    @torch.no_grad()
    def set_kappa(self, value):
        fraction = torch.as_tensor((float(value) - 1.0) / (self.k_max - 1.0)).clamp(1e-6, 1 - 1e-6)
        self.kappa_logit.fill_(float(torch.logit(fraction)))

    def forward(self, x):
        raw = super().forward(x)
        k = self.k_max
        logits = raw[..., 5 * k:]
        if self.training:
            # log of the UNnormalised weights softmax(logits) * gate; kappa_nll reads them as log-weights directly
            pi_part = torch.log_softmax(logits, dim=-1) + log_gate(self.kappa, self.tau, k)
        else:
            pi_part = torch.where(hard_open(self.kappa.detach(), k), logits, torch.full_like(logits, SENTINEL))
        return torch.cat([raw[..., :5 * k], pi_part], dim=-1)

    def soft_open_count(self):
        """Mean over forecast steps of the soft number of open components (differentiable in kappa)."""
        j = component_index(self.k_max, self.kappa)
        return torch.sigmoid((self.kappa[:, None] - j[None, :] + 0.5) / self.tau).sum(-1).mean()
