"""Split-and-grow MDN trainer. K grows from initial_k to k_max, one split per phase.

Smoke runs are limited to <= 6 epochs; full training requires --full and
explicit user approval (see AGENTS.md).
"""
import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import (capture_rng_state, restore_checkpoint, restore_rng_state,  # noqa: E402
                              set_global_seed, utc_now)
from utils.mdn_distribution import build_mdn_distribution  # noqa: E402
from eval import MDN_Forecaster  # noqa: E402
from growing_mdn import base_hashes  # noqa: E402
from growing_mdn.artifacts import ARCH, GrowingTracker  # noqa: E402
from growing_mdn.model import build_model, grow_model, num_components  # noqa: E402
from growing_mdn.optim_state import remap_adam_state  # noqa: E402
from growing_mdn.selection import component_scores, pick_split, source_sigma_median  # noqa: E402

GROWTH_KEYS = ('initial_k', 'k_max', 'phase0_epochs', 'phase_epochs',
               'split_delta_relative', 'restart_lr_factor', 'max_split_nll_change')


def check_config(cfg):
    if cfg.model_params.get('mdn_parameterization', {'mode': 'legacy'}) != {'mode': 'legacy'}:
        raise ValueError('Baseline covariance policy must stay legacy')
    if cfg.train_params['eval_data_reduction'] != 1 or cfg.train_params['dynamic_input_horizon']:
        raise ValueError('Use full deterministic validation and a fixed observation horizon')
    growth = cfg.experiment_params.get('growth', {})
    missing = [key for key in GROWTH_KEYS if key not in growth]
    if missing:
        raise ValueError(f'Missing growth settings: {missing}')
    if growth['k_max'] <= growth['initial_k'] or growth['initial_k'] < 1:
        raise ValueError('Need 1 <= initial_k < k_max')
    if not 0 < growth['split_delta_relative'] <= 0.5:
        raise ValueError('split_delta_relative must be in (0, 0.5]')
    if not 0 < growth['restart_lr_factor'] <= 1:
        raise ValueError('restart_lr_factor must be in (0, 1]')
    if not growth['max_split_nll_change'] > 0:
        raise ValueError('max_split_nll_change must be positive')
    if growth['initial_k'] != cfg.model_params['num_gaussians']:
        raise ValueError('model_params.num_gaussians must equal growth.initial_k')
    total = growth['phase0_epochs'] + (growth['k_max'] - growth['initial_k']) * growth['phase_epochs']
    if total != cfg.train_params['train_epochs']:
        raise ValueError(f'train_epochs {cfg.train_params["train_epochs"]} != phase total {total}')
    return growth


def phase_epochs(growth, phase):
    return growth['phase0_epochs'] if phase == 0 else growth['phase_epochs']


@torch.no_grad()
def validation_nll(model, X, y, device, batch_size, k):
    model.eval()
    total = 0.0
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        raw = model(torch.as_tensor(X[start:stop], dtype=torch.float32, device=device))
        target = torch.as_tensor(y[start:stop], dtype=torch.float32, device=device)
        total += float(-build_mdn_distribution(raw, k).log_prob(target).mean()) * (stop - start)
    return total / len(X)


def phase_lr_scale(growth, phase):
    return 1.0 if phase == 0 else growth['restart_lr_factor']


def make_optimizer(cfg, model, epochs, lr_scale=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_params['lr_default'] * lr_scale)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=cfg.train_params['lr_start_factor'],
        end_factor=cfg.train_params['lr_end_factor'], total_iters=epochs)
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, k, batch, device, tracker, epoch):
    model.train()
    X, y = loader.get_train_data()
    total = 0.0
    for start in range(0, len(X), batch):
        stop = min(start + batch, len(X))
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.as_tensor(X[start:stop], dtype=torch.float32, device=device)
        target = torch.as_tensor(y[start:stop], dtype=torch.float32, device=device)
        try:
            raw = model(inputs)
            nll = -build_mdn_distribution(raw, k).log_prob(target).mean()
            if not torch.isfinite(nll):
                raise FloatingPointError('Non-finite loss')
            nll.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Non-finite gradient')
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            torch.save({'epoch': epoch, 'start': start, 'X': inputs.cpu(), 'y': target.cpu(),
                        'error': str(exc), 'model_state_dict': model.state_dict(),
                        'rng_state': capture_rng_state()}, tracker.run_dir / 'failure.pt')
            tracker.diverged(epoch, epoch - 1)
            raise
        optimizer.step()
        total += float(nll.detach()) * (stop - start)
    return total / len(X), len(X)


def official_metrics(cfg, tracker, loader, model, device, epoch, smoke):
    """Legacy repository metrics; metric RNG must not disturb training RNG."""
    rng = capture_rng_state()
    try:
        set_global_seed(cfg.experiment_params['seed'])
        metric_loader = loader
        if smoke:
            metric_loader = copy.copy(loader)
            fixed = [a[tracker.fixed_indices] for a in loader.eval_data]
            metric_loader.get_eval_data = lambda: fixed
        evaluator = MDN_Forecaster(cfg, model, metric_loader, 'eval', logging.getLogger('growing'), device)
        return evaluator.evaluate(epoch=epoch)
    finally:
        restore_rng_state(rng)


def finish_phase(cfg, tracker, loader, model, optimizer, scheduler, history, device, epoch, smoke):
    """Restore the phase's best state, record official metrics, save the phase checkpoint."""
    k = num_components(model)
    best = torch.load(tracker.checkpoint_dir / f'best_k{k:02d}.pt', map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    optimizer.load_state_dict(best['optimizer_state_dict'])
    scheduler.load_state_dict(best['scheduler_state_dict'])
    val = validation_nll(model, loader.eval_data[0], loader.eval_data[1], device,
                         cfg.test_params['batch_size'], k)
    metrics = official_metrics(cfg, tracker, loader, model, device, epoch, smoke)
    tracker.save_metrics(epoch, metrics, split='fixed_validation_smoke' if smoke else 'eval')
    tracker.phase_complete = True
    tracker.save_checkpoint(f'phase_k{k:02d}', epoch, model, optimizer, scheduler, history)
    tracker.log_phase({'k': k, 'phase': tracker.phase, 'best_epoch': best['epoch'], 'last_epoch': epoch,
                       'validation_nll': val, 'metrics': metrics, 'smoke_metric_subset': bool(smoke),
                       'parameter_count': sum(p.numel() for p in model.parameters()),
                       'timestamp_utc': utc_now()})


def check_split_continuity(before, after, max_change):
    if not (math.isfinite(before) and math.isfinite(after)) or abs(after - before) > max_change:
        raise RuntimeError(f'Split is discontinuous: validation NLL before={before}, after={after}, '
                           f'allowed |change| <= {max_change}')


def split_step(cfg, growth, tracker, loader, model, optimizer, device, epoch):
    """Split the worst-fitting component; return the (K+1) model, optimizer, scheduler."""
    k = num_components(model)
    batch = cfg.test_params['batch_size']
    scores = component_scores(model, loader.train_data[0], loader.train_data[1], device, batch)
    source = pick_split(scores)
    nll_before = validation_nll(model, loader.eval_data[0], loader.eval_data[1], device, batch, k)
    sigma_med = source_sigma_median(model, loader.train_data[0], device, batch, source)
    offsets = growth['split_delta_relative'] * sigma_med
    new_model, rows = grow_model(model, source, offsets)
    nll_after = validation_nll(new_model, loader.eval_data[0], loader.eval_data[1], device, batch, k + 1)
    new_opt, new_sched = make_optimizer(cfg, new_model, growth['phase_epochs'], growth['restart_lr_factor'])
    remap_adam_state(optimizer, new_opt, model, new_model, rows,
                     zero_new_rows=bool(growth.get('zero_new_moments', False)))
    record = {'epoch': epoch, 'k_before': k, 'k_after': k + 1, 'split_component': source,
              'component_scores': [None if not np.isfinite(s) else float(s) for s in scores],
              'validation_nll_before': nll_before, 'validation_nll_after': nll_after,
              'split_delta_relative': growth['split_delta_relative'],
              'offset_abs_min': float(offsets.min()), 'offset_abs_median': float(np.median(offsets)),
              'offset_abs_max': float(offsets.max()), 'aborted': False, 'timestamp_utc': utc_now()}
    try:
        check_split_continuity(nll_before, nll_after, growth['max_split_nll_change'])
    except RuntimeError as exc:
        record['aborted'] = True
        tracker.log_growth(record)
        tracker._write_manifest('aborted_split_discontinuity')
        tracker.event('aborted_split_discontinuity', epoch, {'error': str(exc)})
        raise
    cfg.model_params['num_gaussians'] = k + 1
    tracker.log_growth(record)
    return new_model, new_opt, new_sched


def run(config, run_id, resume=None, smoke=False, stop_after_phase=None, stop_after_epoch=None):
    cfg = ConfigLoader(str(config), 'imptc', False, False, Path(config).stem, 'growing_mdn', 'training')
    growth = check_config(cfg)
    if smoke and cfg.train_params['train_epochs'] > 6:
        raise ValueError('Smoke is limited to <= 6 epochs')
    base_hashes.verify()
    run_dir = Path(cfg.result_path) / 'runs' / run_id
    if run_dir.exists() and not resume:
        raise FileExistsError(run_dir)
    set_global_seed(cfg.experiment_params['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(cfg)
    loader.load_train_data()
    loader.load_eval_data()
    n_phases = growth['k_max'] - growth['initial_k'] + 1
    phase, epoch, epoch_in_phase, k, saved = 0, 0, 0, growth['initial_k'], None
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved.get('architecture') != ARCH or saved['run_id'] != run_id:
            raise ValueError('Incompatible resume checkpoint')
        for key in ('train_params', 'experiment_params'):
            if saved['resolved_config'][key] != getattr(cfg, key):
                raise ValueError(f'Resume config differs: {key}')
        phase, epoch, epoch_in_phase, k = (saved['phase'], saved['epoch'],
                                           saved['epoch_in_phase'], saved['num_gaussians'])
    cfg.model_params['num_gaussians'] = k
    model = build_model(cfg.model_params, k, device)
    optimizer, scheduler = make_optimizer(cfg, model, phase_epochs(growth, phase), phase_lr_scale(growth, phase))
    tracker = GrowingTracker(cfg, loader, device, run_id)
    history = []
    if resume:
        saved = restore_checkpoint(resume, model, optimizer, scheduler, map_location=device)
        if not loader.load_state_dict(saved['data_loader_state']):
            raise ValueError('Checkpoint has no data-loader state')
        history = saved['train_history']
        tracker.restore(saved)
        tracker.event('run_resumed', epoch, {'checkpoint': str(resume)})
    batch = cfg.train_params['batch_size']
    keep_best = bool(resume) and not saved['phase_complete']
    if resume and saved['phase_complete'] and phase == n_phases - 1:
        tracker.save_checkpoint('final', epoch, model, optimizer, scheduler, history)
        tracker.complete(epoch)
        base_hashes.verify()
        return run_dir
    if resume and saved['phase_complete']:
        model, optimizer, scheduler = split_step(cfg, growth, tracker, loader, model, optimizer, device, epoch)
        phase, epoch_in_phase = phase + 1, 0
    while True:
        k = num_components(model)
        tracker.start_phase(phase, k, keep_best=keep_best)
        keep_best = False
        n_epochs = phase_epochs(growth, phase)
        for ep in range(epoch_in_phase + 1, n_epochs + 1):
            started = time.time()
            epoch += 1
            lr = optimizer.param_groups[0]['lr']
            train_nll, n_train = train_epoch(model, loader, optimizer, k, batch, device, tracker, epoch)
            scheduler.step()
            val = validation_nll(model, loader.eval_data[0], loader.eval_data[1], device,
                                 cfg.test_params['batch_size'], k)
            if not np.isfinite(val):
                tracker.diverged(epoch, epoch - 1)
                raise FloatingPointError('Non-finite validation NLL')
            tracker.epoch_in_phase = ep
            row = {'run_id': run_id, 'epoch': epoch, 'train_nll': train_nll, 'validation_nll': val,
                   'learning_rate': lr, 'duration_seconds': time.time() - started,
                   'train_sample_count': n_train, 'validation_sample_count': len(loader.eval_data[0]),
                   'gpu_peak_memory_bytes': int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0,
                   'finite': True, 'timestamp_utc': utc_now(),
                   'phase': phase, 'num_gaussians': k, 'epoch_in_phase': ep}
            history.append(row)
            best = tracker.update_best(epoch, val, model, optimizer, scheduler, history)
            periodic = tracker.should_checkpoint(epoch)
            if periodic:
                tracker.save_checkpoint(f'epoch_{epoch:04d}', epoch, model, optimizer, scheduler, history)
            row.update(is_best=best, checkpoint_saved=bool(periodic or best), full_metrics_evaluated=False)
            tracker.log_epoch(row)
            tracker.save_checkpoint('last', epoch, model, optimizer, scheduler, history, capture_predictions=False)
            print(f'Epoch {epoch} (K={k}, {ep}/{n_epochs}): train NLL={train_nll:.5f}, val NLL={val:.5f}', flush=True)
            if stop_after_epoch is not None and epoch == stop_after_epoch:
                tracker._write_manifest('stopped_for_test')
                base_hashes.verify()
                return run_dir
        finish_phase(cfg, tracker, loader, model, optimizer, scheduler, history, device, epoch, smoke)
        if stop_after_phase is not None and phase == stop_after_phase:
            tracker._write_manifest('stopped_for_test')
            base_hashes.verify()
            return run_dir
        if phase == n_phases - 1:
            break
        model, optimizer, scheduler = split_step(cfg, growth, tracker, loader, model, optimizer, device, epoch)
        phase, epoch_in_phase = phase + 1, 0
    tracker.save_checkpoint('final', epoch, model, optimizer, scheduler, history)
    tracker.complete(epoch)
    base_hashes.verify()
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--stop-after-phase', type=int)
    parser.add_argument('--stop-after-epoch', type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--full', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    name = args.config or ('smoke_growing_peds_imptc.json' if args.smoke else 'growing_peds_imptc.json')
    config = ROOT / 'growing_mdn/configs/imptc' / name
    if args.smoke and json.loads(config.read_text())['train_params']['train_epochs'] > 6:
        raise ValueError('Smoke is limited to <= 6 epochs')
    print(run(config, args.run_id, args.resume, args.smoke, args.stop_after_phase, args.stop_after_epoch))


if __name__ == '__main__':
    main()
