import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import logging
import sys

from base_lstm import LSTM_Trajectory_Forecast, NLL_MDN_loss
from mdn import MDN_Trainer
from utils.config_loader import ConfigLoader
from utils.helper import config_parser,count_model_parameters
from utils.data_loader import DataLoader
from utils.experiment import ExperimentTracker, restore_checkpoint, set_global_seed
from termcolor import colored


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
        saved_params = saved.get('resolved_config', {}).get('model_params')
        if saved_params is not None and saved_params != cfg.model_params:
            raise ValueError('Resume architecture differs from config')

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
    model = LSTM_Trajectory_Forecast(cfg=cfg.model_params).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.train_params['lr_default'])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=cfg.train_params['lr_start_factor'], end_factor=cfg.train_params['lr_end_factor'], total_iters=cfg.train_params['train_epochs'])
    loss_fn = lambda output, target: NLL_MDN_loss(output=output, target=target, num_gaussians=cfg.model_params['num_gaussians'], parameterization=cfg.model_params.get('mdn_parameterization'))
    selected_run = os.environ.get('MDN_RUN_ID')
    if selected_run and not resume_requested and os.path.exists(os.path.join(cfg.result_path, 'runs', selected_run)):
        raise FileExistsError(f'Run already exists: {selected_run}')
    tracker = ExperimentTracker(cfg=cfg, data_loader=data_loader, device=device)

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
    return {'status': 'completed', 'final_epoch': final_epoch}


if __name__ == "__main__":

    # parse arguments
    model_arch = 'stable_mdn'
    type = 'training'
    parser = config_parser()
    args=parser.parse_args()
    if args.run_id:
        os.environ['MDN_RUN_ID'] = args.run_id

    # gpu handling
    if args.gpu:

        gpu_id = args.gpu

    else:

        gpu_id = 0

    # Resolve configs from this source tree, independent of the current directory.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(project_root, model_arch, 'configs', args.target)

    # multiple configs
    if args.configs == 'all':

        configs = [ConfigLoader(config_path=os.path.join(config_dir, conf), target=args.target, with_log=args.log, with_print=args.print, name=conf[:-5], model_arch=model_arch, type=type) for conf in os.listdir(config_dir) if conf.endswith('.json')]

    # single config, selected by user
    else:

        configs = [ConfigLoader(config_path=os.path.join(config_dir, args.configs), target=args.target, with_log=args.log, with_print=args.print, name=args.configs[:-5], model_arch=model_arch, type=type)]

    for cfg in configs:

        training(cfg=cfg, gpu_id=gpu_id)

    sys.exit()
