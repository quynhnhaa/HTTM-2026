"""Sparsemax-MDN training: the baseline `training()` flow with only the model, loss and tracker swapped.

Smoke: --smoke. Full: --full, refused unless sparsemax_mdn/reports/SMOKE.json records a passed review
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
from sparsemax_mdn import base_hashes  # noqa: E402
from sparsemax_mdn.artifacts import ARCH, SparsemaxTracker  # noqa: E402
from sparsemax_mdn.loss import sparsemax_nll  # noqa: E402
from sparsemax_mdn.model import SparsemaxMDN  # noqa: E402

SMOKE_REPORT = ROOT / 'sparsemax_mdn/reports/SMOKE.json'
CONFIG_DIR = ROOT / 'sparsemax_mdn/configs/imptc'
DEFAULT_CONFIGS = {'smoke': 'smoke_sparsemax_k8_peds_imptc.json', 'full': 'sparsemax_k8_peds_imptc.json'}


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

    # Reject incompatible resumes before the tracker writes any run metadata.
    if resume_requested:
        candidate = resume_path or os.path.join(cfg.checkpoint_path, 'model_final.pt')
        if not os.path.isfile(candidate):
            raise FileNotFoundError(candidate)
        from utils.mdn_distribution import checkpoint_parameterization, resolve_parameterization
        saved = torch.load(candidate, map_location='cpu', weights_only=False)
        if checkpoint_parameterization(saved) != resolve_parameterization(cfg.model_params.get('mdn_parameterization')):
            raise ValueError('Cannot resume with a different MDN parameterization; start a new run')
        if saved.get('architecture') != ARCH:
            raise ValueError(f"Cannot resume: checkpoint architecture {saved.get('architecture')!r} != {ARCH!r}")
        saved_params = saved.get('resolved_config', {}).get('model_params')
        if saved_params is not None and saved_params != cfg.model_params:
            raise ValueError('Resume architecture differs from config')

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
    model = SparsemaxMDN(cfg=cfg.model_params).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.train_params['lr_default'])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=cfg.train_params['lr_start_factor'], end_factor=cfg.train_params['lr_end_factor'], total_iters=cfg.train_params['train_epochs'])
    loss_fn = lambda output, target: sparsemax_nll(output, target, cfg.model_params['num_gaussians'])
    selected_run = os.environ.get('MDN_RUN_ID')
    if selected_run and not resume_requested and os.path.exists(os.path.join(cfg.result_path, 'runs', selected_run)):
        raise FileExistsError(f'Run already exists: {selected_run}')
    tracker = SparsemaxTracker(cfg=cfg, data_loader=data_loader, device=device)
    
    # load pretrained model
    if resume_requested:
        
        # check if model exists and load it
        resume_path = resume_path or os.path.join(cfg.checkpoint_path, "model_final.pt")
        if os.path.exists(resume_path):
            
            # yes load it
            checkpoint = restore_checkpoint(resume_path, model, optimizer, scheduler, map_location=device)
            from utils.mdn_distribution import checkpoint_parameterization, resolve_parameterization
            if checkpoint_parameterization(checkpoint) != resolve_parameterization(cfg.model_params.get('mdn_parameterization')):
                raise ValueError('Cannot resume with a different MDN parameterization; start a new run')
            saved_model_params = checkpoint.get('resolved_config', {}).get('model_params')
            if saved_model_params is not None and saved_model_params != cfg.model_params:
                raise ValueError('Resume checkpoint model_params do not match the active config')
            data_state_restored = data_loader.load_state_dict(checkpoint.get('data_loader_state'))
            history = checkpoint.get('train_history', [])
            epoch = int(checkpoint['epoch']) + 1
            tracker.best_epoch = checkpoint.get('best_epoch')
            tracker.best_validation_nll = checkpoint.get('best_validation_nll', float('inf'))
            tracker._write_manifest('running', resumed_from=os.path.abspath(resume_path))
            tracker.event('run_resumed', checkpoint['epoch'], {
                'checkpoint': os.path.abspath(resume_path),
                'data_loader_state_restored': data_state_restored,
            })
            if not data_state_restored:
                warning = 'Checkpoint has no data_loader_state; continuation is valid but not order-exact.'
                if cfg.with_print: print(colored(warning, 'yellow'))
                if cfg.with_log: train_logger.warning(warning)
            if cfg.with_print: print(colored(f"Resume training on gpu: {gpu_id} from {resume_path} at epoch {epoch}", 'green'))
            if cfg.with_log: train_logger.info(f"Resume training on gpu: {gpu_id} from {resume_path} at epoch {epoch}")
            
        # does not exist
        else:
            
            # stop
            if cfg.with_print: print(colored(f"Error: Try to resume training, but pretrained model does not exist, please check path: {os.path.join(cfg.checkpoint_path, 'model_final.pt')}", 'red'))
            if cfg.with_log: train_logger.info(f"Error: Try to resume training, but pretrained model does not exist, please check path: {os.path.join(cfg.checkpoint_path, 'model_final.pt')}")
            if cfg.with_print: print(colored(f"Training finished...", 'red'))
            if cfg.with_log: train_logger.info(f"Training finished...")
            return -1
        
    # train from scratch
    else:
        
        if cfg.with_print: (colored(f"Start training from scratch for: \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - type: {cfg.type} \n - gpu: {gpu_id}", 'green'))
        if cfg.with_log: train_logger.info(f"Start training from scratch for: \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - type: {cfg.type} \n - gpu: {gpu_id}")
        
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
                       name=name[:-5], model_arch='sparsemax_mdn', type='training')
    if cfg.model_params.get('mdn_parameterization', {'mode': 'legacy'}) != {'mode': 'legacy'}:
        raise ValueError('Sparsemax MDN requires the legacy MDN parameterization')
    print(training(cfg=cfg, gpu_id=args.gpu))


if __name__ == '__main__':
    main()
