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
from sparsemax_mdn import base_hashes  # noqa: E402
from sparsemax_mdn.artifacts import ARCH  # noqa: E402

CONFIG_NAME = 'smoke_sparsemax_k8_peds_imptc'
RUNS = ROOT / 'results/trained_models/sparsemax_mdn/imptc' / CONFIG_NAME / 'runs'
METRIC_KEYS = ('ravg_percent', 'rmin_percent', 's68_m2_per_s', 's95_m2_per_s',
               'asaee_m_per_s', 'minade20_m', 'minfde20_m')
K = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='sparsemax_smoke_seed2024')
    args = parser.parse_args()
    run = RUNS / args.run_id
    checks = {}

    base_hashes.verify()
    checks['base_hashes_unchanged'] = True

    rows = list(csv.DictReader((run / 'history.csv').open()))
    n = len(rows)
    assert 1 <= n <= 3
    assert [int(r['epoch']) for r in rows] == list(range(1, n + 1))
    for r in rows:
        assert np.isfinite(float(r['train_nll'])) and np.isfinite(float(r['validation_nll']))
        assert 1.0 <= float(r['mean_support_size']) <= K, r
        assert 0.0 <= float(r['fraction_full_support']) <= 1.0
        assert r['dead_components'] != '' and 0 <= int(r['dead_components']) <= K
    checks['history_rows'] = n

    ckpt_dir = run / 'checkpoints'
    names = [f'epoch_{e:04d}' for e in range(1, n + 1)] + ['best', 'last', 'final']
    for name in names:
        cp = torch.load(ckpt_dir / f'{name}.pt', map_location='cpu', weights_only=False)
        assert cp['architecture'] == ARCH, name
        assert 'rng_state' in cp and 'data_loader_state' in cp and cp['optimizer_state_dict']['state'], name
    checks['checkpoints'] = names

    fixed = np.load(run / 'fixed_samples/inputs.npz')
    ids = fixed['sample_ids']
    assert len(ids) == 8
    zero_found, support_rows = False, 0
    pred_names = [f'epoch_{e:04d}' for e in range(1, n + 1)] + ['best', 'final']
    for name in pred_names:
        pred = np.load(run / 'fixed_samples/predictions' / f'{name}.npz')
        np.testing.assert_array_equal(pred['sample_ids'], ids)
        assert pred['pi'].shape == (8, 48, K), pred['pi'].shape
        np.testing.assert_allclose(pred['pi'].sum(-1), 1, atol=1e-5)
        assert pred['support_size'].shape == (8, 48)
        np.testing.assert_array_equal(pred['support_size'], (pred['pi'] > 0).sum(-1))
        for key in ('raw_output', 'mu', 'sigma', 'rho', 'covariance'):
            assert np.isfinite(pred[key]).all(), (name, key)
        zero_found |= bool((pred['pi'] == 0).any())
        support_rows += 1
    checks['prediction_files'] = pred_names

    metric_files = sorted((run / 'metrics').glob('epoch_*.json'))
    assert metric_files, 'no official metric JSON'
    for path in metric_files:
        metrics = json.loads(path.read_text())['metrics']
        for key in METRIC_KEYS:
            assert np.isfinite(metrics[key]), (path.name, key)
    checks['official_metric_files'] = [p.name for p in metric_files]
    checks['official_metric_keys'] = list(METRIC_KEYS)

    base_hashes.verify()
    # Finding, not a gate: a barely trained model may be dense (all weights > 0).
    finding = ('exact zero pi observed in fixed-sample predictions' if zero_found else
               'NO exactly-zero pi in any smoke prediction (model still dense after a few epochs); '
               'zero handling is covered by unit tests, not by this smoke run')
    report = {'status': 'passed', 'run_id': args.run_id, 'epochs': n, 'checks': checks,
              'finding_zero_pi_in_smoke': zero_found, 'finding_text': finding,
              'last_row': rows[-1], 'full_training_started': False,
              'quality_conclusion': 'None; a few-epoch smoke run says nothing about model quality'}
    out = ROOT / 'sparsemax_mdn/reports'
    out.mkdir(exist_ok=True)
    (out / 'SMOKE.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'SMOKE.md').write_text(
        '# Smoke review: sparsemax MDN\n\n'
        f'Passed. {n} CPU epochs, {len(pred_names)} fixed-sample prediction files (8 samples, pi shape (8, 48, {K})), '
        f'checkpoints {", ".join(names)} with architecture `{ARCH}`, official metrics finite for {len(METRIC_KEYS)} keys, '
        'base_mdn hashes unchanged before and after.\n\n'
        f'Finding: {finding}.\n\n'
        'Metrics come from reduced validation data and say nothing about quality. No full training was started.\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
