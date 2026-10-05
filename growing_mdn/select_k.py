"""Choose K from per-phase validation results. Never reads test data."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.config_loader import ConfigLoader  # noqa: E402
from growing_mdn.selection import METRIC_KEYS, select_k  # noqa: E402


def build_records(summary):
    records = []
    for phase in sorted(summary['phases'].values(), key=lambda p: p['k']):
        records.append({'k': phase['k'], 'validation_nll': phase['validation_nll'],
                        **{key: phase['metrics'][key] for key in METRIC_KEYS}})
    return records


def check_run_ready(manifest, resolved_config, summary):
    """Selection needs a completed run with every K present in the phase summary."""
    if manifest.get('status') != 'completed':
        raise ValueError(f"Run is not completed (status={manifest.get('status')})")
    growth = resolved_config['experiment_params']['growth']
    expected = set(range(growth['initial_k'], growth['k_max'] + 1))
    missing = sorted(expected - {int(k) for k in summary['phases']})
    if missing:
        raise ValueError(f'phase_summary.json is missing K values: {missing}')


def check_selection_allowed(run):
    """Selection is frozen once written, and must precede any official test result."""
    run = Path(run)
    if (run / 'selection.json').exists():
        raise ValueError(f"{run / 'selection.json'} already exists; refusing to overwrite a frozen selection")
    official = sorted((run / 'testing').glob('evaluation_k??.json')) if (run / 'testing').exists() else []
    if official:
        raise ValueError(f'Official test result exists, selection must precede it: {[str(o) for o in official]}')


def frozen_rule(resolved_config):
    return resolved_config['experiment_params']['selection']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='growing_peds_imptc.json')
    args = parser.parse_args()
    path = ROOT / 'growing_mdn/configs/imptc' / args.config
    cfg = ConfigLoader(str(path), 'imptc', False, False, path.stem, 'growing_mdn', 'training')
    run = Path(cfg.result_path) / 'runs' / args.run_id
    check_selection_allowed(run)
    summary = json.loads((run / 'phase_summary.json').read_text())
    resolved = json.loads((run / 'resolved_config.json').read_text())
    check_run_ready(json.loads((run / 'run_manifest.json').read_text()), resolved, summary)
    if any(p.get('smoke_metric_subset') for p in summary['phases'].values()):
        raise ValueError('Smoke metrics come from 8 fixed samples; do not select K from them')
    rule = frozen_rule(resolved)
    result = select_k(build_records(summary), rule['epsilon_nll'], rule['ravg_tolerance_pp'],
                      rule['sharpness_tolerance_ratio'])
    result['tolerances'] = dict(rule)
    result.update(run_id=args.run_id, split='validation')
    (run / 'selection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
