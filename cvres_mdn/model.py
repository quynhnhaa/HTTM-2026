"""Baseline LSTM-MDN whose component means are a constant-velocity extrapolation plus a predicted offset.

mu[b, k, i] = raw_mu[b, k, i] + v_last[b] * dt * (k + 1) for every component i (both axes), where v_last is the
velocity in columns `velocity_columns` of the last observed step. No parameter is added: the state dict has the same
keys as the baseline, so the plain baseline class must NOT be used to evaluate a checkpoint of this model.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402


def cv_extrapolation(x, velocity_columns, delta_t, horizon):
    """[B, horizon, 2] constant-velocity positions v_last * dt * (k + 1), k = 0..horizon-1."""
    v_last = x[:, -1, list(velocity_columns)]                                   # [B, 2] in m/s
    steps = torch.arange(1, horizon + 1, dtype=x.dtype, device=x.device)        # [horizon]
    return v_last[:, None, :] * delta_t * steps[None, :, None]


class CVResidualMDN(LSTM_Trajectory_Forecast):
    def __init__(self, cfg):
        super().__init__(cfg)
        settings = cfg.get('cv_residual', {})
        self.velocity_columns = tuple(settings.get('velocity_columns', (2, 3)))
        if len(self.velocity_columns) != 2 or max(self.velocity_columns) >= self.lstm_input_shape:
            raise ValueError(f'invalid velocity_columns {self.velocity_columns}')
        self.delta_t = float(cfg['delta_t'])
        self.k = int(cfg['num_gaussians'])

    def forward(self, x):
        raw = super().forward(x)                                                 # [B, T, 6K]
        k = self.k
        extrap = cv_extrapolation(x, self.velocity_columns, self.delta_t, self.forecast_horizon)  # [B, T, 2]
        shift = torch.cat([extrap[..., 0:1].expand(-1, -1, k), extrap[..., 1:2].expand(-1, -1, k)], dim=-1)
        return torch.cat([raw[..., :2 * k] + shift, raw[..., 2 * k:]], dim=-1)
