"""Carry Adam moments across a head growth so training continues smoothly."""
import torch

from growing_mdn.model import new_component_rows

FC_NAMES = ('fc.weight', 'fc.bias')
ROW_KEYS = ('exp_avg', 'exp_avg_sq')


@torch.no_grad()
def remap_adam_state(old_opt, new_opt, old_model, new_model, rows, zero_new_rows=False):
    old_params = dict(old_model.named_parameters())
    zero_rows = None
    if zero_new_rows:
        k_old = old_model.output_size // 6
        zero_rows = new_component_rows(k_old, old_model.forecast_horizon).to(rows.device)
    for name, new_param in new_model.named_parameters():
        state = old_opt.state.get(old_params[name])
        if not state:
            continue
        mapped = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                mapped[key] = value
            elif name in FC_NAMES and key in ROW_KEYS:
                gathered = value[rows].clone()
                if zero_rows is not None:
                    gathered[zero_rows] = 0
                mapped[key] = gathered
            else:
                mapped[key] = value.clone()
        new_opt.state[new_param] = mapped
