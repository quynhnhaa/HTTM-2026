"""LSTM + MDN head whose mixture weights come from sparsemax instead of softmax."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402
from sparsemax_mdn.sparsemax import sparsemax  # noqa: E402

SENTINEL = -1e9  # finite stand-in for log(0); exp(SENTINEL - max) == 0 exactly in softmax
TINY = 1e-30


class SparsemaxMDN(LSTM_Trajectory_Forecast):
    """Same architecture/initialisation as the baseline; only the pi block is changed.

    Raw output layout stays [B, H, 6K]: mu_x, mu_y, log sigma_x, log sigma_y, rho_pre, then
    log(pi) (pi = sparsemax(logits)) with SENTINEL where pi == 0. The baseline softmax decoder
    therefore returns exactly the sparsemax weights.
    """

    def __init__(self, cfg):
        policy = cfg.get('mdn_parameterization', {'mode': 'legacy'})
        if policy != {'mode': 'legacy'}:
            raise ValueError('SparsemaxMDN requires mdn_parameterization == {"mode": "legacy"}')
        super().__init__(cfg)
        self.k_max = cfg['num_gaussians']

    def forward(self, x):
        raw = super().forward(x)
        k = self.k_max
        pi = sparsemax(raw[..., 5 * k:], dim=-1)
        log_pi = torch.where(pi > 0, torch.log(pi.clamp_min(TINY)), torch.full_like(pi, SENTINEL))
        return torch.cat([raw[..., :5 * k], log_pi], dim=-1)
