"""Tracker that records the pure NLL and the CRPS term next to the baseline artifacts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.experiment import ExperimentTracker  # noqa: E402


class CrpsTracker(ExperimentTracker):
    history_fields = ExperimentTracker.history_fields + ['train_objective', 'train_crps']

    def __init__(self, *args, **kwargs):
        self.stats = None  # LossStats set by the trainer script
        super().__init__(*args, **kwargs)

    def update_best(self, epoch, validation_nll, model, optimizer, scheduler, history):
        # MDN_Trainer appends the epoch row to `history` (train_nll = full objective), then calls this once.
        row = history[-1]
        consumed = self.stats.consume() if self.stats is not None else None
        if consumed:
            row['train_objective'] = row.get('train_nll')
            row['train_nll'] = consumed['train_nll']   # pure NLL, comparable with the baseline
            row['train_crps'] = consumed['train_crps']
        return super().update_best(epoch, validation_nll, model, optimizer, scheduler, history)
