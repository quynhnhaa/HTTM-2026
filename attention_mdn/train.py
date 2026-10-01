"""Train the LSTM temporal-attention MDN in a separate result namespace."""

import os
import sys
from pathlib import Path

import torch
from torch import optim
from torch.optim import lr_scheduler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MDN = PROJECT_ROOT / 'base_mdn'
sys.path.insert(0, str(BASE_MDN))

from mdn import MDN_Trainer  # noqa: E402
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import ExperimentTracker, restore_checkpoint, set_global_seed  # noqa: E402
from utils.helper import config_parser  # noqa: E402

from base_lstm import NLL_MDN_loss  # noqa: E402
from model import AttentionMDN  # noqa: E402


def load_config(args):
    config_path = Path(args.configs)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / 'attention_mdn' / 'configs' / args.target / config_path
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return ConfigLoader(
        config_path=str(config_path), target=args.target,
        with_log=args.log, with_print=args.print,
        name=config_path.stem, model_arch='attention_mdn', type='training',
    )


def loss_fn(output, target, num_gaussians, parameterization=None):
    return NLL_MDN_loss(output, target, num_gaussians, parameterization)


def train(args):
    # Match the baseline's GPU selection and seed ordering.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cfg = load_config(args)
    if cfg.train_params['dynamic_input_horizon']:
        raise ValueError('This controlled comparison requires dynamic_input_horizon=false')
    if cfg.experiment_params.get('loss_name') != 'marginal_nll':
        raise ValueError('Config must identify the baseline marginal NLL')
    if args.resume and not args.run_id:
        raise ValueError('--resume requires --run-id for the existing variant run')
    if not args.resume and args.run_id:
        existing_run = Path(cfg.result_path) / 'runs' / args.run_id
        if existing_run.exists():
            raise FileExistsError(f'Run already exists: {existing_run}')

    if args.resume:
        from utils.mdn_distribution import checkpoint_parameterization, resolve_parameterization
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint_parameterization(saved) != resolve_parameterization(cfg.model_params.get('mdn_parameterization')):
            raise ValueError('Cannot resume with a different MDN parameterization; start a new run')
        if saved.get('resolved_config', {}).get('model_params') != cfg.model_params:
            raise ValueError('Resume architecture differs from config')
        if saved.get('run_id') != args.run_id:
            raise ValueError('Resume run_id differs from --run-id')

    seed = int(cfg.experiment_params.get('seed', 2024))
    set_global_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = DataLoader(cfg=cfg)
    data_loader.load_train_data()
    data_loader.load_eval_data()

    model = AttentionMDN(cfg.model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train_params['lr_default'])
    scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=cfg.train_params['lr_start_factor'],
        end_factor=cfg.train_params['lr_end_factor'],
        total_iters=cfg.train_params['train_epochs'],
    )
    tracker = ExperimentTracker(cfg, data_loader, device, run_id=args.run_id)

    epoch = 1
    history = []
    if args.resume:
        checkpoint = restore_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        from utils.mdn_distribution import checkpoint_parameterization, resolve_parameterization
        if checkpoint_parameterization(checkpoint) != resolve_parameterization(cfg.model_params.get('mdn_parameterization')):
            raise ValueError('Cannot resume with a different MDN parameterization; start a new run')
        saved_cfg = checkpoint.get('resolved_config', {})
        if saved_cfg.get('model_params') != cfg.model_params:
            raise ValueError('Resume checkpoint model_params do not match variant config')
        if saved_cfg.get('experiment_params', {}).get('loss_name') != cfg.experiment_params['loss_name']:
            raise ValueError('Resume checkpoint does not use the baseline marginal NLL')
        if checkpoint.get('run_id') != tracker.run_id:
            raise ValueError('Resume checkpoint run_id does not match --run-id')
        if not data_loader.load_state_dict(checkpoint.get('data_loader_state')):
            raise ValueError('Resume checkpoint lacks data-loader ordering state')
        epoch = int(checkpoint['epoch']) + 1
        history = checkpoint['train_history']
        tracker.best_epoch = checkpoint.get('best_epoch')
        tracker.best_validation_nll = checkpoint.get('best_validation_nll', float('inf'))
        tracker._write_manifest('running', resumed_from=str(Path(args.resume).resolve()))
        tracker.event('run_resumed', checkpoint['epoch'], {'checkpoint': str(args.resume)})

    import logging
    logger = logging.getLogger(f'attention_mdn.{tracker.run_id}')
    logger.setLevel(logging.INFO)
    log_handler = None
    if cfg.with_log:
        log_handler = logging.FileHandler(
            Path(cfg.evaluation_path) / 'training.log', mode='a' if args.resume else 'w'
        )
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(log_handler)

    try:
        trainer = MDN_Trainer(
            cfg=cfg, model=model,
            loss_fn=lambda output, target: loss_fn(output, target, cfg.model_params['num_gaussians'], cfg.model_params.get('mdn_parameterization')),
            optimizer=optimizer, scheduler=scheduler, device=device, epoch=epoch,
            loss_hist=history, logger=logger, tracker=tracker,
        )
        last_epoch, history, diverged = trainer.train(data_loader)
        if diverged:
            tracker.diverged(last_epoch + 1, last_epoch)
            print(f'Attention training diverged; last completed epoch: {last_epoch}')
            return 1

        trainer.save(epoch=last_epoch, diverged=False, final=True)
        tracker.save_checkpoint('final', last_epoch, model, optimizer, scheduler, history)
        tracker.complete(last_epoch)
        print(f'Completed {last_epoch} epochs. Artifacts: {tracker.run_dir}')
        return 0
    finally:
        if log_handler is not None:
            logger.removeHandler(log_handler)
            log_handler.close()


if __name__ == '__main__':
    parser = config_parser()
    parser.set_defaults(configs='attention_peds_imptc.json')
    parser.add_argument('--resume', default=None, help='Variant last.pt or another compatible checkpoint')
    raise SystemExit(train(parser.parse_args()))
