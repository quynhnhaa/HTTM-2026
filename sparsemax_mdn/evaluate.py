"""Read-only evaluation of a sparsemax-MDN checkpoint: exact NLL + the unchanged official metrics.

    python -m sparsemax_mdn.evaluate --run-id ID --split validation [--limit N] [--exact-pi] [--gpu -1]
    python -m sparsemax_mdn.evaluate --run-id ID --split test --confirm-test-once    # once per run

--exact-pi replaces only the distribution builder of the forecaster instance (spec section 8, caveat 1):
base `Categorical(probs=pi)` scores pi == 0 components with weight about 1.2e-7; the exact variant uses
logits = log(pi) with -inf where pi == 0. The default (clampedpi) is the base code applied as-is.
base_mdn is never modified.
"""
import argparse
import json
import logging
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributions as dist

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import set_global_seed  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402
import eval as base_eval  # noqa: E402
from eval import MDN_Forecaster  # noqa: E402
from sparsemax_mdn import base_hashes  # noqa: E402
from sparsemax_mdn.artifacts import ARCH  # noqa: E402
from sparsemax_mdn.loss import TINY, sparsemax_nll  # noqa: E402
from sparsemax_mdn.model import SparsemaxMDN  # noqa: E402

SEED = 2024
SPLITS = ('validation', 'test')
OFFICIAL_KEYS = ('ravg_percent', 'rmin_percent', 's68_m2_per_s', 's95_m2_per_s', 'asaee_m_per_s',
                 'minade20_m', 'minfde20_m')
TEST_CLAIM = 'test_evaluation_started.lock'
EVALUATOR_VERSION = ('sparsemax_mdn.evaluate v1; official metrics via base_mdn MDN_Forecaster.evaluate unchanged '
                     '(legacy bins); exact NLL via sparsemax_mdn.loss on the whole split in one weighted pass')
CONFIG_DIR = ROOT / 'sparsemax_mdn/configs/imptc'
_CHECKPOINT = re.compile(r'^(best|final|last|epoch_\d{4})$')


# ----------------------------------------------------------------------------- pure helpers
def validate_checkpoint(name):
    if not isinstance(name, str) or not _CHECKPOINT.match(name):
        raise ValueError(f'Checkpoint must be best|final|last|epoch_####, got {name!r}')
    return name


def output_name(split, checkpoint, exact_pi, limited):
    return f"{split}_{checkpoint}_{'exactpi' if exact_pi else 'clampedpi'}{'_limited' if limited else ''}.json"


def persample_name(split, checkpoint, exact_pi, limited):
    return output_name(split, checkpoint, exact_pi, limited)[:-len('.json')] + '_persample.npz'


# plotting function in the eval module -> (key under which its per-sample `data` argument is kept, argument name)
CAPTURED_PLOTS = {'plot_sharpness_over_time': ('sharpness_area_m2', 'data'),
                  'plot_aee_over_time': ('aee_m', 'data'),
                  'plot_reliability_calibration': ('reliability_confidence_sets', 'confidence_sets')}


@contextmanager
def capture_plot_inputs(module, store, plots=None):
    """Patch plotting functions of `module` with wrappers that copy their per-sample input into `store`
    and then delegate unchanged (same args, same return). Originals are always restored."""
    plots = CAPTURED_PLOTS if plots is None else plots
    originals = {name: getattr(module, name) for name in plots}

    def make(original, key, arg):
        def wrapper(*args, **kwargs):
            if arg in kwargs:
                store[key] = np.array(kwargs[arg], copy=True)
            elif args:
                store[key] = np.array(args[0], copy=True)
            return original(*args, **kwargs)
        return wrapper

    try:
        for name, (key, arg) in plots.items():
            setattr(module, name, make(originals[name], key, arg))
        yield store
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def sharpness_scores(areas, horizons, dt, forecast_horizon):
    """Per-sample official-style score [N, L]: sum_h(area_h / ((h+1) dt)) / (forecast_horizon dt)."""
    areas = np.asarray(areas, dtype=np.float64)  # [N, L, H]
    scale = 1.0 / ((np.asarray(horizons, dtype=np.float64) + 1.0) * dt)
    return (areas * scale).sum(-1) / (forecast_horizon * dt)


def official_percentile_scores(areas, horizons, dt, forecast_horizon):
    """Exact replica of base vis.plot_sharpness_over_time: mean over the 101-point percentile grid (not over
    samples) of area/((h+1) dt), summed over horizons. Equals the sample mean only up to quantile-grid error."""
    areas = np.asarray(areas, dtype=np.float64)
    percentiles = np.arange(0.0, 1.01, 0.01)
    out = []
    for lv in range(areas.shape[1]):
        total = 0.0
        for i, h in enumerate(horizons):
            q = np.array([np.percentile(areas[:, lv, i], p * 100) for p in percentiles])
            total += np.mean(q / ((h + 1) * dt))
        out.append(total * (1 / (forecast_horizon * dt)))
    return np.array(out)


def sharpness_per_sample_summary(areas, horizons, dt, forecast_horizon, confidence_levels, threshold=20.0):
    scores = sharpness_scores(areas, horizons, dt, forecast_horizon)
    n = scores.shape[0]
    top = max(1, int(np.ceil(0.05 * n)))
    official = official_percentile_scores(areas, horizons, dt, forecast_horizon)
    summary = {}
    for i, level in enumerate(confidence_levels):
        s = np.sort(scores[:, i])
        summary[f's{int(round(level * 100))}'] = {
            'mean': float(s.mean()), 'median': float(np.median(s)), 'p90': float(np.percentile(s, 90)),
            'p99': float(np.percentile(s, 99)), 'p99_9': float(np.percentile(s, 99.9)), 'max': float(s[-1]),
            f'num_above_{threshold:g}': int((s > threshold).sum()),
            'top5pct_share_of_total': float(s[-top:].sum() / s.sum()) if s.sum() > 0 else float('nan'),
            'official_percentile_grid_score': float(official[i]), 'num_samples': int(n)}
    return summary


def existing_test_results(eval_dir):
    """Non-limited test result files (any checkpoint / flag combination) plus the claim lock."""
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        return []
    found = [p for p in eval_dir.glob('test_*.json') if not p.name.endswith('_limited.json')]
    lock = eval_dir / TEST_CLAIM
    return sorted(found + ([lock] if lock.exists() else []))


def check_split_request(split, confirm_test_once, limit, eval_dir=None):
    """Guard run BEFORE any data or checkpoint is read. eval_dir=None checks the arguments only."""
    if split not in SPLITS:  # exact, case-sensitive match; argparse choices enforce the same
        raise ValueError(f'split must be exactly one of {SPLITS}, got {split!r}')
    if limit is not None and limit <= 0:
        raise ValueError('--limit must be positive')
    if split == 'validation':
        if confirm_test_once:
            raise ValueError('--confirm-test-once only applies to --split test')
        return
    if limit is not None:
        raise ValueError('--limit is not allowed on the test split (one full test evaluation per run)')
    if not confirm_test_once:
        raise ValueError('--split test requires --confirm-test-once (one official test evaluation per run)')
    if eval_dir is not None:
        existing = existing_test_results(eval_dir)
        if existing:
            raise ValueError(f'A test evaluation already exists for this run, refusing: {[str(e) for e in existing]}')


def exact_logits(pi):
    """log(pi) with -inf where pi == 0 (clamp before log so no NaN is ever produced)."""
    return torch.where(pi > 0, torch.log(pi.clamp_min(TINY)), torch.full_like(pi, float('-inf')))


def build_exact_distribution(output, num_gaussians, parameterization=None):
    """MixtureSameFamily with exact weights: components with pi == 0 contribute exactly nothing."""
    params = decode_mdn_output(output, num_gaussians, parameterization)
    components = dist.MultivariateNormal(params['mu'], params['covariance'])
    return dist.MixtureSameFamily(dist.Categorical(logits=exact_logits(params['pi'])), components)


class ExactPiMixin:
    """Only build_distribution changes; every other forecaster method is the base code."""

    def build_distribution(self, output, num_gaussians):
        return build_exact_distribution(output, num_gaussians, self.model_params.get('mdn_parameterization'))


class ExactPiForecaster(ExactPiMixin, MDN_Forecaster):
    pass


def support_summary(pi_batches, k_max):
    """Mean K(x,t), fraction of (x,t) at K_max, components never active, over iterables of pi [B,T,K]."""
    total, count, full = 0.0, 0, 0
    active = torch.zeros(k_max, dtype=torch.bool)
    for pi in pi_batches:
        positive = pi > 0
        size = positive.sum(-1)
        total += float(size.sum())
        count += size.numel()
        full += int((size == k_max).sum())
        active |= positive.flatten(0, -2).any(0).cpu()
    return {'mean_support_size': total / count, 'fraction_full_support': full / count,
            'dead_components': int((~active).sum()), 'k_max': int(k_max), 'num_sample_steps': int(count)}


class _FixedData:
    """Hands the forecaster exactly the arrays chosen here (no random validation subset)."""

    def __init__(self, arrays):
        self.arrays = arrays

    def get_eval_data(self):
        return self.arrays

    def get_test_data(self):
        return self.arrays


@torch.no_grad()
def exact_nll_and_support(model, X, y, k_max, input_horizon, batch_size, device):
    """Exact masked NLL (mean over samples and steps) of the split plus support statistics."""
    weighted, n, pis = 0.0, 0, []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size][:, -input_horizon:, :], dtype=torch.float32, device=device)
        yb = torch.as_tensor(y[start:start + batch_size], dtype=torch.float32, device=device)
        raw = model(xb)
        loss, diverged = sparsemax_nll(raw, yb, k_max)
        if diverged:
            raise FloatingPointError('non-finite NLL while evaluating')
        weighted += float(loss) * len(xb)
        n += len(xb)
        pis.append(decode_mdn_output(raw, k_max)['pi'].cpu())
    return weighted / n, support_summary(pis, k_max)


# ----------------------------------------------------------------------------- CLI
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config')
    parser.add_argument('--run-dir', type=Path, help='Explicit run directory, including runs outside the default result root')
    parser.add_argument('--checkpoint', default='best')
    parser.add_argument('--split', required=True, choices=SPLITS)
    parser.add_argument('--exact-pi', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--confirm-test-once', action='store_true')
    parser.add_argument('--gpu', default='-1')
    return parser


def locate_run(run_id, config=None):
    root = ROOT / 'results/trained_models/sparsemax_mdn/imptc'
    config = config[:-5] if config and config.endswith('.json') else config
    matches = [p for p in root.glob(f'*/runs/{run_id}') if config is None or p.parents[1].name == config]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected exactly one run {run_id!r}, found {matches}')
    return matches[0]


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_checkpoint(args.checkpoint)
    check_split_request(args.split, args.confirm_test_once, args.limit)  # arguments only, nothing touched yet
    run = args.run_dir.resolve() if args.run_dir else locate_run(args.run_id, args.config)
    if args.run_dir:
        manifest = json.loads((run / 'run_manifest.json').read_text())
        if manifest['run_id'] != args.run_id:
            raise ValueError('--run-id does not match --run-dir manifest')
    eval_dir = run / 'evaluation'
    check_split_request(args.split, args.confirm_test_once, args.limit, eval_dir)
    base_hashes.verify()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
    eval_dir.mkdir(exist_ok=True)
    if args.split == 'test':  # claim before the first test byte is read; a crash leaves the lock (user decides)
        fd = os.open(eval_dir / TEST_CLAIM, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            json.dump({'run_id': args.run_id, 'checkpoint': args.checkpoint, 'exact_pi': args.exact_pi}, f)

    config_name = run.parents[1].name
    if args.run_dir:
        resolved = json.loads((run / 'resolved_config.json').read_text())
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as config_file:
            json.dump(resolved, config_file)
            config_file.flush()
            cfg = ConfigLoader(config_file.name, manifest['dataset'], False, False, config_name,
                               'sparsemax_mdn', 'testing' if args.split == 'test' else 'eval')
    else:
        cfg = ConfigLoader(str(CONFIG_DIR / f'{config_name}.json'), 'imptc', False, False, config_name,
                           'sparsemax_mdn', 'testing' if args.split == 'test' else 'eval')
    # Known NumPy containers in locally generated RNG/data-loader state. Keep
    # weights_only enabled instead of allowing arbitrary pickle execution.
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct, np.ndarray, np.dtype,
                                           np.dtypes.UInt32DType, np.dtypes.Float64DType, np.dtypes.Int64DType]):
        saved = torch.load(run / 'checkpoints' / f'{args.checkpoint}.pt', map_location='cpu', weights_only=True)
    if saved.get('architecture') != ARCH:
        raise ValueError(f"Unexpected architecture {saved.get('architecture')!r}, expected {ARCH!r}")
    saved_params = saved.get('resolved_config', {}).get('model_params')
    if saved_params is not None and saved_params != cfg.model_params:
        raise ValueError('Checkpoint model_params do not match the config')
    k_max = cfg.model_params['num_gaussians']
    model = SparsemaxMDN(cfg.model_params)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    model.to(device)

    loader = DataLoader(cfg)
    if args.split == 'test':
        loader.load_test_data()
        arrays = tuple(loader.get_test_data())
    else:
        loader.load_eval_data()
        arrays = tuple(loader.eval_data[:5])
        if args.limit is not None:
            arrays = tuple(a[:args.limit] for a in arrays)
    X, y = arrays[0], arrays[1]
    horizon = cfg.test_params['num_input_horizons']

    nll, support = exact_nll_and_support(model, X, y, k_max, horizon, cfg.test_params['batch_size'], device)
    forecaster_cls = ExactPiForecaster if args.exact_pi else MDN_Forecaster
    with tempfile.TemporaryDirectory() as scratch:  # base plots/txt go here, never next to the training outputs
        cfg.evaluation_path = cfg.testing_path = scratch
        set_global_seed(SEED)
        forecaster = forecaster_cls(cfg, model, _FixedData(arrays), 'testing' if args.split == 'test' else 'eval',
                                    logging.getLogger('sparsemax_evaluate'), device)
        captured = {}
        with capture_plot_inputs(base_eval, captured):
            metrics = forecaster.evaluate(epoch=None)

    result = {
        'run_id': args.run_id, 'config': config_name, 'checkpoint': args.checkpoint,
        'checkpoint_epoch': int(saved['epoch']), 'split': args.split, 'num_samples': int(len(X)),
        'flags': {'exact_pi': args.exact_pi, 'limit': args.limit, 'confirm_test_once': args.confirm_test_once},
        'pi_weighting_in_official_metrics': 'exact (zero weight for pi == 0)' if args.exact_pi
        else 'base code as-is: Categorical(probs=pi), zero weights clamped to ~1.19e-7',
        'seed': SEED, 'exact_nll': nll,
        'official_metrics': {k: metrics[k] for k in OFFICIAL_KEYS},
        'official_metrics_extra': {k: v for k, v in metrics.items() if k not in OFFICIAL_KEYS},
        'support_size': support,
        'sharpness_per_sample_summary': sharpness_per_sample_summary(
            captured['sharpness_area_m2'], cfg.test_params['test_horizons'], forecaster.dt,
            forecaster.forecast_horizon, cfg.test_params['confidence_levels']), 'evaluator_version': EVALUATOR_VERSION,
        'note': 'Whole split, not the random 50% validation subset used for training-time metrics.'
        if args.split == 'validation' else 'Whole test split.'}
    limited = args.limit is not None
    np.savez_compressed(
        eval_dir / persample_name(args.split, args.checkpoint, args.exact_pi, limited),
        sample_index=np.arange(len(X)), confidence_levels=np.asarray(cfg.test_params['confidence_levels']),
        horizons=np.asarray(cfg.test_params['test_horizons']), delta_t=float(forecaster.dt),
        forecast_horizon=int(forecaster.forecast_horizon), **captured)
    out = eval_dir / output_name(args.split, args.checkpoint, args.exact_pi, args.limit is not None)
    out.write_text(json.dumps(result, indent=2) + '\n')
    base_hashes.verify()
    print(json.dumps(result, indent=2))
    print(f'wrote {out}')
    return result


if __name__ == '__main__':
    main()
