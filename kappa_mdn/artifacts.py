"""Tracker that records kappa, K(t), tau and the pure NLL / penalty split next to the base artifacts."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.experiment import ExperimentTracker  # noqa: E402
from kappa_mdn.model import hard_k  # noqa: E402

ARCH = 'kappa_mdn_v2'


def tau_at(epoch, tau_start, tau_end, anneal_epochs):
    """Linear schedule: tau_start at epoch 1, tau_end at epoch `anneal_epochs`, constant afterwards."""
    if anneal_epochs <= 1:
        return float(tau_end)
    fraction = min(1.0, max(0.0, (epoch - 1) / (anneal_epochs - 1)))
    return float(tau_start + (tau_end - tau_start) * fraction)


class KappaTracker(ExperimentTracker):
    history_fields = ExperimentTracker.history_fields + [
        'train_objective', 'train_penalty', 'tau', 'kappa_mean', 'kappa_min', 'kappa_max',
        'mean_K', 'fraction_at_k_max', 'soft_open_count']

    def __init__(self, *args, **kwargs):
        self.stats = None  # LossStats set by the trainer script
        super().__init__(*args, **kwargs)

    def update_best(self, epoch, validation_nll, model, optimizer, scheduler, history):
        # MDN_Trainer appends the epoch row to `history`, then calls this once, then log_epoch(row) (same dict).
        row = history[-1]
        row['tau'] = float(model.tau)
        consumed = self.stats.consume() if self.stats is not None else None
        if consumed:
            row['train_objective'] = row.get('train_nll')
            row['train_nll'] = consumed['train_nll']
            row['train_penalty'] = consumed['train_penalty']
        kappa = model.kappa.detach().cpu()
        k_hard = hard_k(kappa, model.k_max)
        row.update(kappa_mean=float(kappa.mean()), kappa_min=float(kappa.min()), kappa_max=float(kappa.max()),
                   mean_K=float(k_hard.mean()), fraction_at_k_max=float((k_hard == model.k_max).float().mean()),
                   soft_open_count=float(model.soft_open_count()))
        # gate softness for the next epoch (saved in the checkpoint buffer, so resume continues the schedule)
        schedule = self.cfg.experiment_params['kappa']
        tau_start = self.cfg.model_params['kappa']['tau_start']
        model.tau.fill_(tau_at(epoch + 1, tau_start, schedule['tau_end'], schedule['anneal_epochs']))
        return super().update_best(epoch, validation_nll, model, optimizer, scheduler, history)

    def checkpoint_payload(self, epoch, model, optimizer, scheduler, history):
        payload = super().checkpoint_payload(epoch, model, optimizer, scheduler, history)
        payload['architecture'] = ARCH
        payload['kappa'] = model.kappa.detach().cpu().tolist()
        payload['tau'] = float(model.tau)
        return payload

    def capture_fixed_predictions(self, model, epoch, label=None):
        super().capture_fixed_predictions(model, epoch, label)
        path = self.prediction_dir / f'{label or f"epoch_{epoch:04d}"}.npz'
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        payload['support_size'] = (payload['pi'] > 0).sum(-1).astype(np.int64)
        payload['kappa'] = model.kappa.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(path, **payload)
