import os
import torch
import logging
from pathlib import Path

from base_lstm import LSTM_Trajectory_Forecast
from eval import MDN_Forecaster
from utils.config_loader import ConfigLoader
from utils.helper import config_parser, count_model_parameters
from utils.data_loader import DataLoader
from termcolor import colored


def resolve_test_checkpoint(cfg, checkpoint=None, run_id=None):
    """Resolve a test checkpoint, defaulting to the validation-best model.

    Resolution order:
    1. Explicit ``--checkpoint`` or ``MDN_TEST_CHECKPOINT``.
    2. ``best.pt`` from ``--run-id`` or ``MDN_RUN_ID``.
    3. The only available run-level ``best.pt`` for this configuration.

    Ambiguous run discovery is rejected so testing never silently evaluates the
    wrong experiment. Legacy ``model_final.pt`` remains usable only by passing
    its path explicitly.
    """
    explicit = checkpoint or os.environ.get('MDN_TEST_CHECKPOINT')
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Explicit test checkpoint does not exist: {path}')
        return path

    selected_run = run_id or os.environ.get('MDN_RUN_ID')
    runs_dir = Path(cfg.result_path) / 'runs'
    if selected_run:
        path = runs_dir / selected_run / 'checkpoints' / 'best.pt'
        if not path.is_file():
            raise FileNotFoundError(
                f'Validation-best checkpoint does not exist for run {selected_run!r}: {path}'
            )
        return path.resolve()

    candidates = sorted(runs_dir.glob('*/checkpoints/best.pt'))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            f'No run-level best.pt found under {runs_dir}. '
            'Pass --checkpoint explicitly for a legacy or final checkpoint.'
        )
    run_names = ', '.join(path.parent.parent.name for path in candidates)
    raise RuntimeError(
        f'Multiple runs contain best.pt ({run_names}). Pass --run-id to select one.'
    )


def testing(args, gpu_id):
    """Run testing framework
    """

    #--- init and setup
    model_arch = 'stable_mdn'
    type = 'testing'

    print(colored(f"Starting testing: on GPU: {gpu_id}", 'green'))

    # Resolve configs from this source tree, independent of the current directory.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(project_root, model_arch, 'configs', args.target)

    # multiple configs
    if args.configs == 'all':

        configs = [ConfigLoader(config_path=os.path.join(config_dir, conf), target=args.target, with_log=args.log, with_print=args.print, name=conf, model_arch=model_arch, type=type) for conf in os.listdir(config_dir) if conf.endswith('.json')]

    # single config, selected by user
    else:

        configs = [ConfigLoader(config_path=os.path.join(config_dir, args.configs), target=args.target, with_log=args.log, with_print=args.print, name=args.configs[:-5], model_arch=model_arch, type=type)]

    # start the testings
    for cfg in configs:
        test_logger = logging.getLogger("stable_mdn.testing")

        # model params
        lstm_input_shape = cfg.model_params["lstm_input_shape"]
        max_input_horizon = cfg.model_params['max_input_horizon']

        # training params
        batch_size = cfg.test_params['batch_size']

        # eval params
        confidence_levels = cfg.test_params['confidence_levels']
        mesh_range_x = cfg.test_params['mesh_range_x']
        mesh_range_y = cfg.test_params['mesh_range_y']
        mesh_resolution = cfg.test_params['mesh_resolution']
        num_samples = cfg.test_params['num_samples']
        plot_examples = cfg.test_params['plot_examples']
        plot_examples_to_map = cfg.test_params['plot_examples_to_map']
        plot_step = cfg.test_params['plot_step']

        # set cuda gpu device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # init dataloader
        data_loader = DataLoader(cfg=cfg)

        # load data
        data_loader.load_test_data()

        # logger
        if cfg.with_log:

            log_file_path = os.path.join(cfg.testing_path, 'testing.log')
            os.remove(log_file_path) if os.path.exists(log_file_path) else None
            test_logger = logging.getLogger('testing')
            test_logger.setLevel(logging.INFO)
            log_file_handler = logging.FileHandler(log_file_path)
            log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            test_logger.addHandler(log_file_handler)

        # Load the validation-best model by default. A final or legacy
        # checkpoint must be requested explicitly with --checkpoint.
        checkpoint_path = resolve_test_checkpoint(
            cfg, checkpoint=args.checkpoint, run_id=args.run_id
        )
        checkpoint = torch.load(f=checkpoint_path, map_location=device, weights_only=False)
        from utils.mdn_distribution import apply_checkpoint_parameterization
        apply_checkpoint_parameterization(cfg, checkpoint)
        saved_model_params = checkpoint.get('resolved_config', {}).get('model_params')
        if saved_model_params is not None and {k:v for k,v in saved_model_params.items() if k != 'mdn_parameterization'} != {k:v for k,v in cfg.model_params.items() if k != 'mdn_parameterization'}:
            raise ValueError(
                f'Checkpoint model_params do not match active config: {checkpoint_path}'
            )
        model = LSTM_Trajectory_Forecast(cfg=cfg.model_params)
        model.load_state_dict(checkpoint["model"])

        # create eval forecaster
        eval_forecaster = MDN_Forecaster(cfg=cfg, model=model, data_loader=data_loader, type='testing', device=device, logger=test_logger)

        if cfg.with_print: print(colored(f"Start testing for: \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - checkpoint: {checkpoint_path} \n - model parameters: {count_model_parameters(model=model)}", 'green'))
        if cfg.with_log: test_logger.info(f"Start testing for: \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - checkpoint: {checkpoint_path} \n - model parameters: {count_model_parameters(model=model)}")

        # Run evaluation tasks
        eval_forecaster.evaluate(epoch=None)

        # save example plots
        if plot_examples or plot_examples_to_map:

            if cfg.with_print: print(colored(f"Plotting {int(len(data_loader.test_data[0])/plot_step)} examples...", 'magenta'))
            if cfg.with_log: test_logger.info(f"Plotting {int(len(data_loader.test_data[0])/plot_step)} examples...")

            eval_forecaster.save_examples(
                epoch=None,
                n_samples=num_samples,
                mesh_range_x=mesh_range_x,
                mesh_range_y=mesh_range_y,
                mesh_resolution=mesh_resolution,
                confidence_levels=confidence_levels,
                plot_ego=plot_examples,
                plot_map=plot_examples_to_map
                )

        if cfg.with_print: print(colored(f"Finished testing...", 'green'))
        if cfg.with_log: test_logger.info(f"Finished testing...")

    if cfg.with_print: print(colored(f"All tests completed, shutdown...", 'green'))
    if cfg.with_log: test_logger.info(f"All tests completed, shutdown...")

    if cfg.with_log:

        log_file_handler.close()
        test_logger.removeHandler(log_file_handler)

    print(colored(f"Finished testing: {cfg.name} on GPU: {gpu_id}", 'cyan'))

    return


if __name__ == "__main__":

    parser = config_parser()
    args=parser.parse_args()

    # gpu handling
    if args.gpu:

        gpu_id = args.gpu

    else:

        gpu_id = 0

    testing(args=args, gpu_id=gpu_id)
