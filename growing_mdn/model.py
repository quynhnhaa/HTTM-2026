"""LSTM-MDN whose output head can grow by splitting one component (K -> K+1).

The architecture and covariance parameterization are exactly those of
``base_mdn/base_lstm.py``; only the construction of a (K+1)-component head from
a K-component one is added here.
"""
import copy
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402

# mu_x, mu_y, log sigma_x, log sigma_y, rho_pre, pi_logit
BLOCKS = 6
MU_BLOCKS = (0, 1)
PI_BLOCK = 5


def build_model(model_params, k, device='cpu'):
    params = copy.deepcopy(model_params)
    params['num_gaussians'] = int(k)
    if params['output_factor'] != BLOCKS:
        raise ValueError('growing_mdn assumes the 6-block legacy MDN layout')
    model = LSTM_Trajectory_Forecast(params).to(device)
    model.growth_params = params
    return model


def num_components(model):
    return model.output_size // BLOCKS


def fc_row_map(k_old, source, horizon):
    """For every row of the (K+1)-head, the row of the K-head it copies.

    Flat row index = t * 6K + block * K + k. The appended component (index
    ``k_old``) copies component ``source``.
    """
    k_new = k_old + 1
    t = torch.arange(horizon).view(-1, 1, 1)
    b = torch.arange(BLOCKS).view(1, -1, 1)
    k = torch.arange(k_new).view(1, 1, -1)
    old_k = torch.where(k == k_old, torch.full_like(k, source), k)
    return (t * BLOCKS * k_old + b * k_old + old_k).reshape(-1)


def new_component_rows(k_old, horizon):
    """Rows of the (K+1)-head that belong to the appended component."""
    k_new = k_old + 1
    t = torch.arange(horizon).view(-1, 1, 1)
    b = torch.arange(BLOCKS).view(1, -1, 1)
    return (t * BLOCKS * k_new + b * k_new + k_old).reshape(-1)


def _expand_offsets(offsets, horizon):
    """Positive scalar -> constant [horizon, 2]; otherwise validate a [horizon, 2] array."""
    tensor = torch.as_tensor(offsets, dtype=torch.float32).detach().cpu()
    if tensor.ndim == 0:
        tensor = tensor.expand(horizon, len(MU_BLOCKS)).clone()
    if tuple(tensor.shape) != (horizon, len(MU_BLOCKS)):
        raise ValueError(f'offsets must be a scalar or shape {(horizon, len(MU_BLOCKS))}, got {tuple(tensor.shape)}')
    if not bool(torch.isfinite(tensor).all()) or not bool((tensor > 0).all()):
        raise ValueError('offsets must be finite and positive')
    return tensor


@torch.no_grad()
def grow_model(model, source, offsets):
    """Split component ``source`` into two; return ``(new_model, rows)``.

    ``offsets``: positive scalar or absolute [horizon, 2] bias offsets for mu_x, mu_y.
    The source copy moves by +offsets, the new component by -offsets.
    """
    k_old = num_components(model)
    if not 0 <= source < k_old:
        raise ValueError(f'source {source} outside [0, {k_old})')
    horizon = model.forecast_horizon
    device = model.fc.weight.device
    offsets = _expand_offsets(offsets, horizon).to(device)
    new = build_model(model.growth_params, k_old + 1, device=device)
    new.lstm.load_state_dict(model.lstm.state_dict())
    rows = fc_row_map(k_old, source, horizon).to(device)
    new.fc.weight.copy_(model.fc.weight[rows])
    new.fc.bias.copy_(model.fc.bias[rows])
    bias = new.fc.bias.view(horizon, BLOCKS, k_old + 1)
    bias[:, PI_BLOCK, source] -= math.log(2.0)
    bias[:, PI_BLOCK, k_old] -= math.log(2.0)
    for axis, block in enumerate(MU_BLOCKS):
        bias[:, block, source] += offsets[:, axis]
        bias[:, block, k_old] -= offsets[:, axis]
    return new, rows
