"""Reproducible experiment artifacts without changing the baseline model/loss."""

import csv
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from utils.mdn_distribution import decode_mdn_output


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    # torch.load(map_location='cuda') also maps this serialized CPU RNG tensor;
    # PyTorch's CPU generator explicitly requires a CPU ByteTensor.
    torch.set_rng_state(state['torch_cpu'].cpu())
    if torch.cuda.is_available() and state.get('torch_cuda'):
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state['torch_cuda']])


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class ExperimentTracker:
    history_fields = [
        'run_id', 'epoch', 'train_nll', 'validation_nll', 'learning_rate',
        'duration_seconds', 'train_sample_count', 'validation_sample_count',
        'gpu_peak_memory_bytes', 'is_best', 'checkpoint_saved',
        'full_metrics_evaluated', 'finite', 'timestamp_utc'
    ]

    def __init__(self, cfg, data_loader, device, run_id=None):
        self.cfg = cfg
        self.data_loader = data_loader
        self.device = device
        self.params = cfg.experiment_params
        self.seed = int(self.params.get('seed', 2024))
        self.run_id = run_id or os.environ.get('MDN_RUN_ID') or (
            datetime.now().strftime('%Y%m%d-%H%M%S') + f'_{cfg.target}_{cfg.name}_seed{self.seed}'
        )
        self.run_dir = Path(cfg.result_path) / 'runs' / self.run_id
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.fixed_dir = self.run_dir / 'fixed_samples'
        self.prediction_dir = self.fixed_dir / 'predictions'
        self.metric_dir = self.run_dir / 'metrics'
        for path in (self.run_dir, self.checkpoint_dir, self.fixed_dir, self.prediction_dir, self.metric_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.history_path = self.run_dir / 'history.csv'
        self.events_path = self.run_dir / 'events.jsonl'
        self.best_validation_nll = float('inf')
        self.best_epoch = None
        self.fixed_indices = []
        self.fixed_sample_ids = []
        self._write_json(self.run_dir / 'resolved_config.json', self._resolved_config())
        self._write_json(self.run_dir / 'environment.json', self._environment())
        self._write_manifest('running')
        self._prepare_fixed_samples()
        self.event('run_started', None, {'seed': self.seed})

    def _resolved_config(self):
        return {
            'paths': self.cfg.paths,
            'model_params': self.cfg.model_params,
            'train_params': self.cfg.train_params,
            'test_params': self.cfg.test_params,
            'eval_metrics': self.cfg.eval_metrics,
            'experiment_params': self.cfg.experiment_params,
        }

    def _environment(self):
        return {
            'python': sys.version,
            'pytorch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'cuda_driver_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'os': platform.platform(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        }

    def _write_manifest(self, status, **updates):
        path = self.run_dir / 'run_manifest.json'
        manifest = {}
        if path.exists():
            manifest = json.loads(path.read_text())
        manifest.update({
            'schema_version': '1.0', 'run_id': self.run_id, 'status': status,
            'experiment_name': self.params.get('experiment_name', 'official_repo_baseline'),
            'dataset': self.cfg.target,
            'mdn_parameterization': self.cfg.model_params.get('mdn_parameterization', {'mode': 'legacy'}),
            'run_seed': self.seed, 'best_epoch': self.best_epoch,
            'best_validation_nll': None if self.best_epoch is None else self.best_validation_nll,
            'updated_at': utc_now(),
        })
        manifest.update(updates)
        self._write_json(path, manifest)

    @staticmethod
    def _write_json(path, value):
        tmp = Path(str(path) + '.tmp')
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        os.replace(tmp, path)

    def event(self, event, epoch, payload):
        with self.events_path.open('a') as stream:
            stream.write(json.dumps({'timestamp_utc': utc_now(), 'run_id': self.run_id,
                                     'event': event, 'epoch': epoch, 'payload': payload}) + '\n')

    def _prepare_fixed_samples(self):
        manifest_name = self.params.get('fixed_sample_manifest')
        if manifest_name:
            manifest_path = Path(self.cfg.project_root) / manifest_name
            manifest = json.loads(manifest_path.read_text())
        else:
            count = int(self.params.get('fixed_sample_count', 0))
            if count <= 0:
                raise ValueError(
                    'experiment_params must define fixed_sample_manifest or a positive '
                    'fixed_sample_count'
                )
            if count > len(self.data_loader.sample_keys['eval']):
                raise ValueError(
                    f'Cannot select {count} fixed samples from only '
                    f"{len(self.data_loader.sample_keys['eval'])} validation samples"
                )
            rng = np.random.default_rng(self.seed)
            indices = sorted(rng.choice(
                len(self.data_loader.sample_keys['eval']), size=count, replace=False
            ).tolist())
            manifest = {
                'schema_version': '1.0',
                'dataset': self.cfg.target,
            'mdn_parameterization': self.cfg.model_params.get('mdn_parameterization', {'mode': 'legacy'}),
                'split': 'validation',
                'selection_seed': self.seed,
                'selection': 'numpy.default_rng choice without replacement, sorted',
                'samples': [],
            }
            sample_prefix = self.params.get('fixed_sample_id_prefix', self.cfg.name)
            for order, index in enumerate(indices, start=1):
                manifest['samples'].append({
                    'sample_id': f'{sample_prefix}_val_{order:02d}',
                    'pickle_key': self.data_loader.sample_keys['eval'][index],
                    'validation_index': index,
                    'x_sha256': sha256_array(self.data_loader.eval_data[0][index]),
                    'y_sha256': sha256_array(self.data_loader.eval_data[1][index]),
                    'source': str(self.data_loader.eval_data[4][index]),
                })
        key_to_index = {key: idx for idx, key in enumerate(self.data_loader.sample_keys['eval'])}
        for sample in manifest['samples']:
            index = key_to_index[sample['pickle_key']]
            X = self.data_loader.eval_data[0][index]
            y = self.data_loader.eval_data[1][index]
            if sha256_array(X) != sample['x_sha256'] or sha256_array(y) != sample['y_sha256']:
                raise ValueError(f"Fixed sample checksum mismatch: {sample['sample_id']}")
            self.fixed_indices.append(index)
            self.fixed_sample_ids.append(sample['sample_id'])
        self._write_json(self.fixed_dir / 'manifest.json', manifest)
        idx = self.fixed_indices
        np.savez_compressed(
            self.fixed_dir / 'inputs.npz',
            sample_ids=np.asarray(self.fixed_sample_ids),
            X=self.data_loader.eval_data[0][idx].astype(np.float32),
            y=self.data_loader.eval_data[1][idx].astype(np.float32),
            reference_position=self.data_loader.eval_data[2][idx].astype(np.float32),
            rotation_angle=self.data_loader.eval_data[3][idx].astype(np.float32),
            source=self.data_loader.eval_data[4][idx].astype(str),
        )

    def capture_fixed_predictions(self, model, epoch, label=None):
        model_was_training = model.training
        model.eval()
        X = torch.as_tensor(self.data_loader.eval_data[0][self.fixed_indices], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            raw = model(X)
            decoded = decode_mdn_output(raw, self.cfg.model_params['num_gaussians'], self.cfg.model_params.get('mdn_parameterization'))
        name = label or f'epoch_{epoch:04d}'
        np.savez_compressed(
            self.prediction_dir / f'{name}.npz',
            schema_version='1.0', run_id=self.run_id, checkpoint_name=name,
            epoch=np.int64(epoch), sample_ids=np.asarray(self.fixed_sample_ids),
            raw_output=raw.cpu().numpy().astype(np.float32),
            pi=decoded['pi'].cpu().numpy().astype(np.float32),
            mu=decoded['mu'].cpu().numpy().astype(np.float32),
            sigma=decoded['sigma'].cpu().numpy().astype(np.float32),
            rho=decoded['rho'].cpu().numpy().astype(np.float32),
            covariance=decoded['covariance'].cpu().numpy().astype(np.float32),
        )
        if model_was_training:
            model.train()

    def checkpoint_payload(self, epoch, model, optimizer, scheduler, history):
        return {
            'schema_version': '1.0', 'run_id': self.run_id, 'epoch': epoch,
            'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'train_history': history,
            'best_epoch': self.best_epoch, 'best_validation_nll': self.best_validation_nll,
            'resolved_config': self._resolved_config(), 'rng_state': capture_rng_state(),
            'data_loader_state': self.data_loader.state_dict(),
            'created_at': utc_now(),
            # Backward-compatible aliases used by the upstream testing script.
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'loss': [row['train_nll'] if isinstance(row, dict) else row for row in history],
        }

    def save_checkpoint(self, name, epoch, model, optimizer, scheduler, history, capture_predictions=True):
        path = self.checkpoint_dir / f'{name}.pt'
        tmp = Path(str(path) + '.tmp')
        torch.save(self.checkpoint_payload(epoch, model, optimizer, scheduler, history), tmp)
        os.replace(tmp, path)
        if capture_predictions:
            self.capture_fixed_predictions(model, epoch, label=name)
        self.event('checkpoint_saved', epoch, {'name': name, 'path': str(path)})
        return path

    def should_checkpoint(self, epoch):
        explicit = set(self.params.get('checkpoint_epochs', [1, 5, 10]))
        every = int(self.params.get('checkpoint_every', 100))
        return epoch in explicit or (every > 0 and epoch % every == 0)

    def should_evaluate_metrics(self, epoch):
        every = int(self.params.get('metric_every', 250))
        return every > 0 and epoch % every == 0

    def log_epoch(self, row):
        exists = self.history_path.exists()
        with self.history_path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.history_fields)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in self.history_fields})

    def save_metrics(self, epoch, metrics, split='eval'):
        snapshot = {
            'schema_version': '1.0', 'run_id': self.run_id, 'split': split,
            'checkpoint': f'epoch_{epoch:04d}', 'epoch': epoch,
            'evaluation_seed': self.seed,
            'test_horizons': self.cfg.test_params['test_horizons'],
            'times_seconds': [(step + 1) * self.cfg.model_params['delta_t'] for step in self.cfg.test_params['test_horizons']],
            'num_k_samples': self.cfg.test_params['num_k_samples'],
            'confidence_mc_samples': self.cfg.test_params['num_samples'],
            'sharpness_formula_version': 'corrected',
            'metrics': metrics,
        }
        self._write_json(self.metric_dir / f'epoch_{epoch:04d}.json', snapshot)
        history_path = self.metric_dir / 'history.csv'
        flat = {'run_id': self.run_id, 'epoch': epoch, **metrics}
        existing = []
        fields = list(flat)
        if history_path.exists():
            with history_path.open() as stream:
                reader = csv.DictReader(stream)
                existing = list(reader)
                fields = list(dict.fromkeys((reader.fieldnames or []) + fields))
        with history_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(existing)
            writer.writerow(flat)
        self.event('evaluation_completed', epoch, {'metrics': metrics})

    def update_best(self, epoch, validation_nll, model, optimizer, scheduler, history):
        if validation_nll < self.best_validation_nll:
            self.best_validation_nll = float(validation_nll)
            self.best_epoch = int(epoch)
            self.save_checkpoint('best', epoch, model, optimizer, scheduler, history)
            self._write_manifest('running')
            return True
        return False

    def complete(self, epoch):
        self._write_manifest('completed', final_epoch=epoch, finished_at=utc_now())
        self.event('run_completed', epoch, {})

    def diverged(self, failed_epoch, last_completed_epoch):
        self._write_manifest(
            'diverged',
            diverged_at_epoch=int(failed_epoch),
            last_completed_epoch=int(last_completed_epoch),
            finished_at=utc_now(),
        )
        self.event('run_diverged', failed_epoch, {
            'last_completed_epoch': int(last_completed_epoch),
            'best_epoch': self.best_epoch,
            'best_validation_nll': self.best_validation_nll,
        })


def restore_checkpoint(path, model, optimizer=None, scheduler=None, restore_rng=True, map_location='cpu'):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint['model']))
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', checkpoint['optimizer']))
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if restore_rng and 'rng_state' in checkpoint:
        restore_rng_state(checkpoint['rng_state'])
    return checkpoint
