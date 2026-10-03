"""Fork a completed stable run; preserve parent best and reset only LR schedule."""
import argparse
import copy
import importlib.util
import json
import logging
import os
from pathlib import Path

import torch

from base_lstm import LSTM_Trajectory_Forecast, NLL_MDN_loss
from mdn import MDN_Trainer
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.experiment import ExperimentTracker, restore_checkpoint, set_global_seed
from utils.mdn_distribution import checkpoint_parameterization

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    parent = Path(args.parent_run).resolve()
    manifest = json.loads((parent / 'run_manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('Parent run must be completed')
    saved_cfg = json.loads((parent / 'resolved_config.json').read_text())
    config = copy.deepcopy(saved_cfg)
    final_epoch = int(manifest['final_epoch'])
    config['train_params']['train_epochs'] = final_epoch + args.extra_epochs
    config['train_params']['resume_training'] = False
    config['experiment_params'].update(
        experiment_name=args.run_id, parent_run=str(parent),
        continuation_start_epoch=final_epoch + 1,
        continuation_lr_start=args.lr_start, continuation_lr_end=args.lr_end,
        continuation_schedule_epochs=500,
    )
    if args.smoke:
        config['train_params'].update(train_data_reduction=.001, batch_size=128)
        config['experiment_params'].update(metric_every=0, checkpoint_every=1)
    arch = 'stable_attention_mdn' if args.attention else 'stable_mdn'
    name = parent.parent.parent.name
    expected = ROOT / 'results/trained_models' / arch / 'imptc' / name / 'runs' / args.run_id
    if expected.exists():
        raise FileExistsError(expected)
    # Config snapshot is separate from the reviewed full configs.
    config_path = ROOT / arch / 'configs/imptc' / (args.run_id + '.json')
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    cfg = ConfigLoader(str(config_path), 'imptc', True, True, name, arch, 'training')
    seed = int(cfg.experiment_params['seed'])
    set_global_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(cfg)
    loader.load_train_data()
    loader.load_eval_data()
    if args.attention:
        spec = importlib.util.spec_from_file_location('continuation_attention', ROOT / 'stable_attention_mdn/model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.AttentionMDN(cfg.model_params).to(device)
    else:
        model = LSTM_Trajectory_Forecast(cfg.model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_params['lr_default'])
    checkpoint = restore_checkpoint(parent / 'checkpoints/final.pt', model, optimizer, map_location=device)
    if checkpoint_parameterization(checkpoint) != cfg.model_params['mdn_parameterization']:
        raise ValueError('Distribution policy differs')
    if int(checkpoint['epoch']) != final_epoch:
        raise ValueError('Parent final epoch differs')
    if not loader.load_state_dict(checkpoint['data_loader_state']):
        raise ValueError('Parent lacks loader ordering')
    # Keep Adam moments/step counts; start a fresh 500-step linear LR schedule.
    for group in optimizer.param_groups:
        group['lr'] = args.lr_start
        group['initial_lr'] = args.lr_start
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1., end_factor=args.lr_end / args.lr_start,
        total_iters=500,
    )
    tracker = ExperimentTracker(cfg, loader, device, run_id=args.run_id)
    best = torch.load(parent / 'checkpoints/best.pt', map_location=device, weights_only=False)
    tracker.best_epoch = int(best['epoch'])
    tracker.best_validation_nll = float(best['best_validation_nll'])
    # Preserve complete parent-best training state; mark its source explicitly.
    inherited = dict(best)
    inherited.update(run_id=args.run_id, parent_run=str(parent),
                     inherited_checkpoint=str(parent / 'checkpoints/best.pt'))
    torch.save(inherited, tracker.checkpoint_dir / 'best.pt')
    model.load_state_dict(best['model_state_dict'])
    tracker.capture_fixed_predictions(model, tracker.best_epoch, label='best')
    model.load_state_dict(checkpoint['model_state_dict'])
    tracker.capture_fixed_predictions(model, final_epoch, label='parent_final')
    history = checkpoint['train_history']
    tracker._write_manifest('running', parent_run=str(parent),
                            continuation_start_epoch=final_epoch + 1,
                            lr_schedule={'start': args.lr_start, 'end': args.lr_end, 'epochs': 500})
    tracker.event('continuation_started', final_epoch, {
        'parent_final': str(parent / 'checkpoints/final.pt'),
        'parent_best': str(parent / 'checkpoints/best.pt'),
        'adam_state_preserved': True, 'scheduler_reset': True,
    })
    logger = logging.getLogger(args.run_id)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(Path(cfg.evaluation_path) / 'training.log')
    logger.addHandler(handler)
    try:
        trainer = MDN_Trainer(cfg, model,
            lambda out, y: NLL_MDN_loss(out, y, cfg.model_params['num_gaussians'], cfg.model_params['mdn_parameterization']),
            optimizer, scheduler, final_epoch + 1, history, logger, device, tracker)
        epoch, history, diverged = trainer.train(loader)
        if diverged:
            tracker.diverged(epoch + 1, epoch)
            return 1
        trainer.save(epoch, False, True)
        tracker.save_checkpoint('final', epoch, model, optimizer, scheduler, history)
        tracker.complete(epoch)
        print('Continuation completed:', tracker.run_dir)
        return 0
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--attention', action='store_true')
    p.add_argument('--extra-epochs', type=int, default=500)
    p.add_argument('--lr-start', type=float, default=1e-5)
    p.add_argument('--lr-end', type=float, default=1e-7)
    p.add_argument('--gpu', default='0')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if not (0 < args.extra_epochs <= 500 and 0 < args.lr_end <= args.lr_start):
        p.error('Need 1..500 extra epochs and 0 < lr-end <= lr-start')
    raise SystemExit(main(args))
