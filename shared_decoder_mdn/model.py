"""LSTM encoder (identical to the baseline) + a decoder shared across the 48 forecast steps.

Baseline: one linear layer maps the 8-dim LSTM state to all 48 * 6K mixture parameters, i.e. every forecast step
has its own weights. Here a small MLP maps [LSTM state, time features of step t] to the 6K parameters of step t and
its weights are shared by all steps. The time features are fixed (not learned): the normalised step tau = t/(T-1)
and sin/cos(pi * f * tau) for f = 1..F, so the parameters vary smoothly with t by construction.
The raw output keeps the baseline layout [B, T, 6K] (blocks of K: mu_x, mu_y, log sigma_x, log sigma_y, rho_pre,
pi_logit), so the baseline loss, decoder and evaluator work unchanged.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402


def time_features(horizon, num_frequencies):
    """[T, 1 + 2F] fixed features of the forecast step."""
    tau = torch.linspace(0.0, 1.0, horizon)[:, None]
    freqs = torch.arange(1, num_frequencies + 1, dtype=torch.float32)[None, :]
    return torch.cat([tau, torch.sin(math.pi * freqs * tau), torch.cos(math.pi * freqs * tau)], dim=-1)


class SharedDecoderMDN(LSTM_Trajectory_Forecast):
    def __init__(self, cfg):
        super().__init__(cfg)
        del self.fc  # replaced by the shared decoder
        settings = cfg['decoder']
        hidden, frequencies = int(settings['hidden']), int(settings['num_frequencies'])
        features = time_features(self.forecast_horizon, frequencies)
        self.register_buffer('step_features', features)
        self.decoder = nn.Sequential(
            nn.Linear(self.lstm_hidden_size + features.shape[1], hidden), nn.Tanh(),
            nn.Linear(hidden, self.output_size))
        for layer in self.decoder:  # same convention as the baseline fc: Xavier weights, zero biases
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        state = lstm_out[:, -1, :]                                            # [B, H_lstm]
        steps = self.forecast_horizon
        joint = torch.cat([state[:, None, :].expand(-1, steps, -1),
                           self.step_features[None].expand(state.shape[0], -1, -1)], dim=-1)
        return self.decoder(joint)                                            # [B, T, 6K]
