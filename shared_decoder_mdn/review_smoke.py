"""Review the smoke run of shared_decoder_mdn and write reports/SMOKE.json (gate for --full)."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared_decoder_mdn import base_hashes  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else 'smoke_shared_decoder_peds_imptc'
RUN = ROOT / 'results/trained_models/shared_decoder_mdn/imptc' / NAME / 'runs' / (sys.argv[2] if len(sys.argv) > 2 else 'sd_smoke_seed2024')
OUT = ROOT / 'shared_decoder_mdn/reports/SMOKE.json'
FIELDS = ('epoch', 'train_nll', 'validation_nll', 'learning_rate')


def main():
    checks = {}
    rows = list(csv.DictReader(open(RUN / 'history.csv')))
    assert len(rows) == 3, len(rows)
    for r in rows:
        assert all(math.isfinite(float(r[k])) for k in FIELDS), r
    checks['history_rows_finite'] = len(rows)
    for name in ('best', 'last', 'final', 'epoch_0001', 'epoch_0002', 'epoch_0003'):
        cp = torch.load(RUN / 'checkpoints' / f'{name}.pt', map_location='cpu', weights_only=False)
        assert 'decoder.0.weight' in cp['model_state_dict'] and 'fc.weight' not in cp['model_state_dict'], name
    checks['checkpoints'] = 'best,last,final,epoch_0001-3 contain the shared decoder and no fc'
    metrics = json.loads((RUN / 'metrics/epoch_0003.json').read_text())['metrics']
    keys = ('ravg_percent', 'rmin_percent', 's68_m2_per_s', 's95_m2_per_s', 'asaee_m_per_s', 'minade20_m', 'minfde20_m')
    assert all(math.isfinite(float(metrics[k])) for k in keys), metrics
    checks['official_metrics_finite'] = list(keys)
    preds = sorted((RUN / 'fixed_samples/predictions').glob('*.npz'))
    assert preds, 'no fixed-sample predictions'
    with np.load(preds[0]) as d:
        assert 'pi' in d.files and 'mu' in d.files, d.files
    checks['fixed_sample_files'] = len(preds)
    base_hashes.verify()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'status': 'passed', 'checks': checks, 'full_training_started': False,
                               'quality_conclusion': 'None; a 3-epoch smoke run says nothing about model quality'},
                              indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
