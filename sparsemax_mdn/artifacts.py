"""Tracker adding support-size statistics next to the base artifacts."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.experiment import ExperimentTracker  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402

ARCH = 'sparsemax_mdn_v1'
STATS_SUBSET = 2048


@torch.no_grad()
def support_statistics(model, X, k_max, device, batch_size=512):
    """Mean K(x,t), fraction of (x,t) with full support, number of dead components."""
    was_training = model.training
    model.eval()
    sizes, active = [], torch.zeros(k_max, dtype=torch.bool)
    for start in range(0, len(X), batch_size):
        raw = model(torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device))
        pi = decode_mdn_output(raw, k_max)['pi']
        positive = pi > 0
        sizes.append(positive.sum(-1).flatten().cpu())
        active |= positive.flatten(0, -2).any(0).cpu()
    if was_training:
        model.train()
    sizes = torch.cat(sizes).float()
    return {'mean_support_size': float(sizes.mean()),
            'fraction_full_support': float((sizes == k_max).float().mean()),
            'dead_components': int((~active).sum())}


class SparsemaxTracker(ExperimentTracker):
    history_fields = ExperimentTracker.history_fields + [
        'mean_support_size', 'fraction_full_support', 'dead_components']

    def __init__(self, *args, **kwargs):
        self._support_stats = {}
        super().__init__(*args, **kwargs)

    def update_best(self, epoch, validation_nll, model, optimizer, scheduler, history):
        # Called once per epoch by MDN_Trainer before log_epoch: compute stats on the fixed subset.
        X = self.data_loader.eval_data[0][:STATS_SUBSET]
        self._support_stats = support_statistics(
            model, X, self.cfg.model_params['num_gaussians'], self.device)
        return super().update_best(epoch, validation_nll, model, optimizer, scheduler, history)

    def log_epoch(self, row):
        row = {**row, **self._support_stats}
        super().log_epoch(row)

    def checkpoint_payload(self, epoch, model, optimizer, scheduler, history):
        payload = super().checkpoint_payload(epoch, model, optimizer, scheduler, history)
        payload['architecture'] = ARCH
        return payload

    def capture_fixed_predictions(self, model, epoch, label=None):
        super().capture_fixed_predictions(model, epoch, label)
        path = self.prediction_dir / f'{label or f"epoch_{epoch:04d}"}.npz'
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        payload['support_size'] = (payload['pi'] > 0).sum(-1).astype(np.int64)
        np.savez_compressed(path, **payload)
