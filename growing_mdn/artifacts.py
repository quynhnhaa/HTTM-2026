"""Tracker that records K, phase and growth events next to the base artifacts."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.experiment import ExperimentTracker  # noqa: E402

ARCH = 'growing_mdn_v1'


class GrowingTracker(ExperimentTracker):
    history_fields = ExperimentTracker.history_fields + ['phase', 'num_gaussians', 'epoch_in_phase']

    def __init__(self, *args, **kwargs):
        # Set before the base constructor, which may already touch tracker state.
        self.phase = 0
        self.current_k = None
        self.epoch_in_phase = 0
        self.phase_best = float('inf')
        self.phase_best_epoch = None
        self.phase_complete = False
        super().__init__(*args, **kwargs)

    def start_phase(self, phase, k, keep_best=False):
        self.phase, self.current_k = int(phase), int(k)
        if not keep_best:
            self.epoch_in_phase = 0
            self.phase_best = float('inf')
            self.phase_best_epoch = None
            self.phase_complete = False

    def update_best(self, epoch, validation_nll, model, optimizer, scheduler, history):
        """Per-phase best; the run-level best fields track the minimum over all phases."""
        if validation_nll >= self.phase_best:
            return False
        self.phase_best = float(validation_nll)
        self.phase_best_epoch = int(epoch)
        if validation_nll < self.best_validation_nll:
            self.best_validation_nll = float(validation_nll)
            self.best_epoch = int(epoch)
        self.save_checkpoint(f'best_k{self.current_k:02d}', epoch, model, optimizer, scheduler, history)
        self._write_manifest('running')
        return True

    def checkpoint_payload(self, epoch, model, optimizer, scheduler, history):
        payload = super().checkpoint_payload(epoch, model, optimizer, scheduler, history)
        payload.update(architecture=ARCH, phase=self.phase, num_gaussians=self.current_k,
                       epoch_in_phase=self.epoch_in_phase, phase_best_nll=self.phase_best,
                       phase_best_epoch=self.phase_best_epoch, phase_complete=self.phase_complete)
        return payload

    def restore(self, saved):
        self.phase, self.current_k = saved['phase'], saved['num_gaussians']
        self.epoch_in_phase = saved['epoch_in_phase']
        self.phase_best = saved['phase_best_nll']
        self.phase_best_epoch = saved['phase_best_epoch']
        self.phase_complete = saved['phase_complete']
        self.best_epoch = saved.get('best_epoch')
        self.best_validation_nll = saved.get('best_validation_nll', float('inf'))

    def capture_fixed_predictions(self, model, epoch, label=None):
        super().capture_fixed_predictions(model, epoch, label)
        path = self.prediction_dir / f'{label or f"epoch_{epoch:04d}"}.npz'
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        payload.update(phase=np.int64(self.phase), num_gaussians=np.int64(self.current_k))
        np.savez_compressed(path, **payload)

    def log_growth(self, record):
        with (self.run_dir / 'growth_history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        self.event('component_split', record.get('epoch'), record)

    def log_phase(self, record):
        """phase_summary.json keyed by K; rewritten atomically so a rerun is idempotent."""
        path = self.run_dir / 'phase_summary.json'
        summary = json.loads(path.read_text()) if path.exists() else {'phases': {}}
        summary['phases'][str(record['k'])] = record
        self._write_json(path, summary)
        self.event('phase_completed', record.get('last_epoch'), {'k': record['k']})
