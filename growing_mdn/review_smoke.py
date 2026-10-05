"""Review smoke artifacts: pipeline integrity only, not model quality."""
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
from growing_mdn import base_hashes  # noqa: E402
from growing_mdn.artifacts import ARCH  # noqa: E402
from growing_mdn.selection import METRIC_KEYS  # noqa: E402

CONFIG = ROOT / 'growing_mdn/configs/imptc/smoke_growing_peds_imptc.json'
RUNS = ROOT / 'results/trained_models/growing_mdn/imptc/smoke_growing_peds_imptc/runs'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='growing_smoke_seed2024')
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    g = cfg['experiment_params']['growth']
    split_guard = g['max_split_nll_change']
    ks = list(range(g['initial_k'], g['k_max'] + 1))
    expected_k_per_epoch = [g['initial_k']] * g['phase0_epochs'] + [k for k in ks[1:] for _ in range(g['phase_epochs'])]
    run = RUNS / args.run_id

    base_hashes.verify()
    rows = list(csv.DictReader((run / 'history.csv').open()))
    assert [int(r['epoch']) for r in rows] == list(range(1, len(expected_k_per_epoch) + 1))
    assert [int(r['num_gaussians']) for r in rows] == expected_k_per_epoch
    assert all(np.isfinite(float(r['train_nll'])) and np.isfinite(float(r['validation_nll'])) for r in rows)

    growth = [json.loads(line) for line in (run / 'growth_history.jsonl').read_text().splitlines()]
    assert [(r['k_before'], r['k_after']) for r in growth] == [(k, k + 1) for k in ks[:-1]]
    deltas = {f"{r['k_before']}->{r['k_after']}": r['validation_nll_after'] - r['validation_nll_before'] for r in growth}
    assert all(r['aborted'] is False for r in growth), growth
    assert all(np.isfinite(d) and abs(d) <= split_guard for d in deltas.values()), deltas

    summary = json.loads((run / 'phase_summary.json').read_text())['phases']
    assert sorted(int(k) for k in summary) == ks
    for k in ks:
        item = summary[str(k)]
        assert np.isfinite(item['validation_nll']) and item['smoke_metric_subset'] is True
        for key in METRIC_KEYS:
            assert np.isfinite(item['metrics'][key]), (k, key)

    fixed = np.load(run / 'fixed_samples/inputs.npz')
    assert len(fixed['sample_ids']) == 8
    for k in ks:
        for name in (f'best_k{k:02d}', f'phase_k{k:02d}'):
            cp = torch.load(run / 'checkpoints' / f'{name}.pt', map_location='cpu', weights_only=False)
            assert cp['architecture'] == ARCH and cp['num_gaussians'] == k
            assert 'rng_state' in cp and 'data_loader_state' in cp and cp['optimizer_state_dict']['state']
        pred = np.load(run / 'fixed_samples/predictions' / f'phase_k{k:02d}.npz')
        np.testing.assert_array_equal(pred['sample_ids'], fixed['sample_ids'])
        assert pred['mu'].shape == (8, 48, k, 2) and pred['pi'].shape == (8, 48, k)
        assert int(pred['num_gaussians']) == k
        np.testing.assert_allclose(pred['pi'].sum(-1), 1, atol=1e-6)
        for key in ('raw_output', 'sigma', 'rho', 'covariance'):
            assert np.isfinite(pred[key]).all()
    for name in ('last', 'final'):
        assert (run / 'checkpoints' / f'{name}.pt').exists()

    # Resume checks (run by the task-6 procedure); all runs must exist and be verified.
    def history(name):
        path = RUNS / name / 'history.csv'
        if not path.exists():
            raise FileNotFoundError(f'Missing resume evidence: {path}')
        return [(r['epoch'], r['num_gaussians'], float(r['validation_nll'])) for r in csv.DictReader(path.open())]

    ref = history('resume_ref')
    resume = {}
    for name in ('resume_cut', 'resume_e1', 'resume_e2', 'resume_p2'):
        cut = history(name)
        same = [a[:2] for a in ref] == [b[:2] for b in cut]
        diff = max(abs(a[2] - b[2]) for a, b in zip(ref, cut)) if same else None
        resume[name] = {'epochs': len(cut), 'verified': bool(same and diff < 1e-6), 'max_abs_validation_nll_diff': diff}
    failed = [n for n, v in resume.items() if not v['verified']]
    if failed:
        raise AssertionError(f'Resume checks failed: {failed} {resume}')
    status = 'passed' if all(v['verified'] for v in resume.values()) and len(resume) == 4 else 'failed'

    report = {'status': status, 'run_id': args.run_id, 'k_sequence': ks, 'epochs': len(rows),
              'fixed_samples': 8, 'base_source_hashes_unchanged': True,
              'split_validation_nll_deltas': deltas,
              'resume_checks_vs_resume_ref': resume,
              'metric_scope': '8 fixed validation samples, MC64, grid 1 m; not full validation metrics',
              'full_training_started': False,
              'quality_conclusion': 'None; a few-epoch smoke run says nothing about the best K'}
    out = ROOT / 'growing_mdn/reports'
    out.mkdir(exist_ok=True)
    (out / 'SMOKE.json').write_text(json.dumps(report, indent=2) + '\n')
    resume_lines = ''.join(
        f"- `{n}`: {'verified' if v['verified'] else 'FAILED'} ({v['epochs']} epochs, max |validation NLL diff| vs resume_ref = {v['max_abs_validation_nll_diff']})\n"
        for n, v in resume.items())
    delta_text = ', '.join(f'{k}: {v:+.4f}' for k, v in deltas.items())
    (out / 'SMOKE.md').write_text(
        '# Smoke review: growing MDN\n\n'
        f'{status.capitalize()}. K sequence {ks}, {len(rows)} epochs, 8 fixed validation samples, base_mdn hashes unchanged. '
        'Metrics come from the 8 fixed samples only and are not full validation results. '
        'No conclusion about the best K and no full training was started.\n\n'
        '## Split continuity\n\n'
        f'Validation NLL change across each split: {delta_text}. The review uses only a loose sanity guard '
        f'(finite and |delta| <= {split_guard}, the smoke config max_split_nll_change). The strict continuity check on trained weights is in '
        'SPLIT_CHECK.json and the regression unit test; real-data deltas of a barely trained model are data-dependent.\n\n'
        '## Resume checks (CPU, compared with uninterrupted `resume_ref`, tolerance 1e-6)\n\n'
        'resume_cut: stop after phase 0, resume from phase_k01.pt. resume_e1: mid-phase resume from last.pt after epoch 1. '
        'resume_e2: resume from last.pt after epoch 2 (phase-end work not yet run). resume_p2: stop after final phase, resume from phase_k03.pt.\n\n'
        + resume_lines)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
