"""Review smoke artifacts (fail closed): pipeline integrity only, not model quality."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'base_mdn'))
from kappa_mdn import base_hashes  # noqa: E402
from kappa_mdn.artifacts import ARCH, tau_at  # noqa: E402

CONFIG_NAME = 'smoke_kappa_k16_peds_imptc'
CONFIG = ROOT / 'kappa_mdn/configs/imptc' / f'{CONFIG_NAME}.json'
RUNS = ROOT / 'results/trained_models/kappa_mdn/imptc' / CONFIG_NAME / 'runs'
METRIC_KEYS = ('ravg_percent', 'rmin_percent', 's68_m2_per_s', 's95_m2_per_s',
               'asaee_m_per_s', 'minade20_m', 'minfde20_m')
K, T = 16, 48


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='kappa_smoke_seed2024')
    args = parser.parse_args()
    run = RUNS / args.run_id
    cfg = json.loads(CONFIG.read_text())
    schedule, kappa_cfg = cfg['experiment_params']['kappa'], cfg['model_params']['kappa']
    base_lr, multiplier = cfg['train_params']['lr_default'], schedule['lr_multiplier']
    checks = {}

    base_hashes.verify()
    checks['base_hashes_unchanged'] = True

    rows = list(csv.DictReader((run / 'history.csv').open()))
    n = len(rows)
    assert 1 <= n <= 3
    assert [int(r['epoch']) for r in rows] == list(range(1, n + 1))
    for r in rows:
        for key in ('train_nll', 'validation_nll', 'train_penalty', 'train_objective', 'kappa_mean', 'mean_K', 'tau'):
            assert np.isfinite(float(r[key])), (key, r)
        # objective = pure NLL + penalty (penalty is positive and the NLL column is the pure one)
        assert float(r['train_penalty']) > 0
        assert abs(float(r['train_objective']) - (float(r['train_nll']) + float(r['train_penalty']))) < 1e-3 * (1 + abs(float(r['train_objective']))), r
        assert 1.0 <= float(r['mean_K']) <= K
        assert 0.0 <= float(r['fraction_at_k_max']) <= 1.0
    # tau used in epoch e must follow the declared schedule
    for r in rows:
        expected = tau_at(int(r['epoch']), kappa_cfg['tau_start'], schedule['tau_end'], schedule['anneal_epochs'])
        assert abs(float(r['tau']) - expected) < 1e-6, (r['epoch'], r['tau'], expected)
    checks['history_rows'] = n

    ckpt_dir = run / 'checkpoints'
    names = [f'epoch_{e:04d}' for e in range(1, n + 1)] + ['best', 'last', 'final']
    for name in names:
        cp = torch.load(ckpt_dir / f'{name}.pt', map_location='cpu', weights_only=False)
        assert cp['architecture'] == ARCH, name
        assert 'rng_state' in cp and 'data_loader_state' in cp and cp['optimizer_state_dict']['state'], name
        assert len(cp['kappa']) == T and 'kappa_logit' in cp['model_state_dict'] and 'tau' in cp['model_state_dict'], name
        groups = cp['optimizer_state_dict']['param_groups']
        assert len(groups) == 2 and abs(groups[1]['lr'] / groups[0]['lr'] - multiplier) < 1e-6, [g['lr'] for g in groups]
    final = torch.load(ckpt_dir / 'final.pt', map_location='cpu', weights_only=False)
    moved = float(torch.tensor(final['kappa']).sub(kappa_cfg['init']).abs().max())
    assert moved > 0, 'kappa never moved from its initial value'
    checks['checkpoints'] = names
    checks['kappa_max_move_from_init'] = moved
    checks['optimizer_groups_lr_ratio'] = multiplier

    fixed = np.load(run / 'fixed_samples/inputs.npz')
    ids = fixed['sample_ids']
    assert len(ids) == 8
    pred_names = [f'epoch_{e:04d}' for e in range(1, n + 1)] + ['best', 'final']
    for name in pred_names:
        pred = np.load(run / 'fixed_samples/predictions' / f'{name}.npz')
        np.testing.assert_array_equal(pred['sample_ids'], ids)
        assert pred['pi'].shape == (8, T, K), pred['pi'].shape
        np.testing.assert_allclose(pred['pi'].sum(-1), 1, atol=1e-5)
        k_hard = np.clip(np.round(pred['kappa']), 1, K).astype(int)
        expected_support = np.broadcast_to(k_hard[None, :], (8, T))
        np.testing.assert_array_equal(pred['support_size'], expected_support)   # exact zeros above round(kappa)
        np.testing.assert_array_equal(pred['support_size'], (pred['pi'] > 0).sum(-1))
        for key in ('raw_output', 'mu', 'sigma', 'rho', 'covariance', 'kappa'):
            assert np.isfinite(pred[key]).all(), (name, key)
    checks['prediction_files'] = pred_names

    metric_files = sorted((run / 'metrics').glob('epoch_*.json'))
    assert metric_files, 'no official metric JSON'
    for path in metric_files:
        metrics = json.loads(path.read_text())['metrics']
        for key in METRIC_KEYS:
            assert np.isfinite(metrics[key]), (path.name, key)
    checks['official_metric_files'] = [p.name for p in metric_files]

    base_hashes.verify()
    report = {'status': 'passed', 'run_id': args.run_id, 'epochs': n, 'checks': checks, 'last_row': rows[-1],
              'full_training_started': False,
              'quality_conclusion': 'None; a few-epoch smoke run says nothing about model quality'}
    out = ROOT / 'kappa_mdn/reports'
    out.mkdir(exist_ok=True)
    (out / 'SMOKE.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'SMOKE.md').write_text(
        '# Smoke review: kappa MDN\n\n'
        f'Passed. {n} CPU epochs, {len(pred_names)} fixed-sample prediction files (8 samples), checkpoints '
        f'{", ".join(names)} with architecture `{ARCH}`, tau followed the declared schedule, the penalty/NLL split is '
        f'consistent, kappa moved by up to {moved:.4f} from its initial value, optimizer lr ratio {multiplier} for kappa, '
        f'exact-zero weights above round(kappa), official metrics finite for {len(METRIC_KEYS)} keys, '
        'base_mdn hashes unchanged before and after.\n\n'
        'Metrics come from reduced validation data and say nothing about quality. No full training was started.\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
