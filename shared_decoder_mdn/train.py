"""Shared-decoder MDN training: the baseline `training()` flow with only the model swapped (loss, Adam, schedule, tracker are the baseline ones).

Smoke: --smoke. Full: --full, refused unless shared_decoder_mdn/reports/SMOKE.json records a passed review
(and, per AGENTS.md, only on explicit user request).
"""
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from mdn import MDN_Trainer  # noqa: E402
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.helper import count_model_parameters  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import restore_checkpoint, set_global_seed  # noqa: E402
from termcolor import colored  # noqa: E402
from shared_decoder_mdn import base_hashes  # noqa: E402
from shared_decoder_mdn.model import SharedDecoderMDN  # noqa: E402
from base_lstm import NLL_MDN_loss  # noqa: E402
from utils.experiment import ExperimentTracker  # noqa: E402

SMOKE_REPORT = ROOT / 'shared_decoder_mdn/reports/SMOKE.json'
CONFIG_DIR = ROOT / 'shared_decoder_mdn/configs/imptc'
DEFAULT_CONFIGS = {'smoke': 'smoke_shared_decoder_peds_imptc.json', 'full': 'shared_decoder_peds_imptc.json'}


def training(cfg, gpu_id):
    """Run training framework
    """
    
    # start the training
    print(colored(f"Starting training: {cfg.name} on GPU: {gpu_id}", 'green'))
    
    # CUDA visibility must be selected before the first CUDA API call.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    seed = int(cfg.experiment_params.get('seed', 2024))
    set_global_seed(seed)
    resume_path = os.environ.get('MDN_RESUME_CHECKPOINT')
    resume_requested = bool(resume_path) or cfg.train_params['resume_training']

    if resume_requested:
        raise NotImplementedError('shared_decoder_mdn does not support resume; start a new run')

    base_hashes.verify()

    # init dataloader
    data_loader = DataLoader(cfg=cfg)
    
    # load data
    data_loader.load_train_data()
    data_loader.load_eval_data()
    
    # logger
    train_logger = None
    log_file_handler = None
    if cfg.with_log:
        
        log_file_path = os.path.join(cfg.evaluation_path, 'training.log')
        train_logger = logging.getLogger('training')
        train_logger.setLevel(logging.INFO)
        log_file_handler = logging.FileHandler(log_file_path, mode='a' if resume_requested else 'w')
        log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        train_logger.addHandler(log_file_handler)
        
    else:
        
        log_file_path = None
    
    #--- init network
    # mixture density network
    epoch = 1
    history = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SharedDecoderMDN(cfg=cfg.model_params).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.train_params['lr_default'])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=cfg.train_params['lr_start_factor'], end_factor=cfg.train_params['lr_end_factor'], total_iters=cfg.train_params['train_epochs'])
    loss_fn = lambda output, target: NLL_MDN_loss(output=output, target=target, num_gaussians=cfg.model_params['num_gaussians'], parameterization=cfg.model_params.get('mdn_parameterization'))
    selected_run = os.environ.get('MDN_RUN_ID')
    if selected_run and not resume_requested and os.path.exists(os.path.join(cfg.result_path, 'runs', selected_run)):
        raise FileExistsError(f'Run already exists: {selected_run}')
    tracker = ExperimentTracker(cfg=cfg, data_loader=data_loader, device=device)
    
    if cfg.with_log: train_logger.info(f'Start training from scratch: {cfg.name} on gpu {gpu_id}')

    #--- start training
    # build trainer
    trainer = MDN_Trainer(cfg=cfg, model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, device=device, epoch=epoch, loss_hist=history, logger=train_logger, tracker=tracker)
    final_epoch, history, diverged = trainer.train(data_loader=data_loader)

    if diverged:
        tracker.diverged(final_epoch + 1, final_epoch)
        message = (
            f'Training diverged during epoch {final_epoch + 1}. '
            f'Preserved last.pt at epoch {final_epoch} and best.pt at epoch '
            f'{tracker.best_epoch}; no final.pt was created.'
        )
        if cfg.with_print: print(colored(message, 'red'))
        if cfg.with_log: train_logger.error(message)
        if cfg.with_log:
            log_file_handler.close()
            train_logger.removeHandler(log_file_handler)
        base_hashes.verify()
        return {'status': 'diverged', 'last_completed_epoch': final_epoch}
    
    # Save final model
    trainer.save(epoch=final_epoch, diverged=False, final=True)
    tracker.save_checkpoint('final', final_epoch, model, optimizer, scheduler, history)
    tracker.complete(final_epoch)
        
    if cfg.with_print: print(colored(f"Saved final model with {count_model_parameters(model=model)} parameters", 'green'))
    if cfg.with_log: train_logger.info(f"Saved final model with {count_model_parameters(model=model)} parameters")
    if cfg.with_print: print(colored(f"All trainings completed, shutdown...", 'green'))
    if cfg.with_log: train_logger.info(f"All trainings completed, shutdown...")
    
    if cfg.with_log:
        
        log_file_handler.close()
        train_logger.removeHandler(log_file_handler)
        
        
    print(colored(f"Finished training: {cfg.name} on GPU: {gpu_id}", 'cyan'))
    base_hashes.verify()
    return {'status': 'completed', 'final_epoch': final_epoch}
    
    
def check_full_allowed():
    if not SMOKE_REPORT.exists() or json.loads(SMOKE_REPORT.read_text()).get('status') != 'passed':
        raise RuntimeError(f'--full refused: no passed smoke review at {SMOKE_REPORT}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--full', action='store_true')
    args = parser.parse_args()
    if args.full:
        check_full_allowed()
    name = args.config or DEFAULT_CONFIGS['smoke' if args.smoke else 'full']
    if not name.endswith('.json'):
        name += '.json'
    config_path = CONFIG_DIR / name
    if args.smoke and json.loads(config_path.read_text())['train_params']['train_epochs'] > 3:
        raise ValueError('Smoke is limited to <= 3 epochs')
    os.environ['MDN_RUN_ID'] = args.run_id
    cfg = ConfigLoader(config_path=str(config_path), target='imptc', with_log=True, with_print=True,
                       name=name[:-5], model_arch='shared_decoder_mdn', type='training')
    print(training(cfg=cfg, gpu_id=args.gpu))


if __name__ == '__main__':
    main()
