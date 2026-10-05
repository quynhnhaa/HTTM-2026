"""Prove base_mdn is untouched: SHA256 snapshot recorded once, verified per run."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'crps_mdn/reports/BASE_SOURCE_HASHES.json'


def compute():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'base_mdn').rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def verify(expected=None):
    expected = json.loads(SNAPSHOT.read_text()) if expected is None else expected
    actual = compute()
    changed = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
    if changed:
        raise RuntimeError(f'base_mdn differs from snapshot: {changed}')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.write:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(compute(), indent=2, sort_keys=True) + '\n')
        print(f'wrote {SNAPSHOT}')
    else:
        verify()
        print('base_mdn unchanged')


if __name__ == '__main__':
    main()
