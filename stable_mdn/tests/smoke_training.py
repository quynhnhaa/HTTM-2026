"""Explicit short IMPTC smoke, including resume; not an auto-discovered test."""
import csv
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stable_mdn'))
from train import training
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.mdn_distribution import build_mdn_distribution
from base_lstm import LSTM_Trajectory_Forecast
from eval import MDN_Forecaster


def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / 'base_mdn').rglob('*') if p.is_file()}


def equal(x, y):
    if isinstance(x, torch.Tensor):
        return torch.equal(x, y)
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
    if isinstance(x, (tuple, list)):
        return len(x) == len(y) and all(equal(a, b) for a, b in zip(x, y))
    return x == y


def main():
    before = hashes()
    path = ROOT / 'stable_mdn/configs/imptc/smoke_stable_peds_imptc.json'
    config = json.loads((path.parent / 'stable_peds_imptc.json').read_text())
    config['train_params'].update(train_epochs=3, batch_size=128, train_data_reduction=.001)
    config['experiment_params'].update(checkpoint_epochs=[1, 2, 3], checkpoint_every=0, metric_every=1)
    config['eval_metrics'].update(sharpness=False, asaee=False)
    config['test_params']['num_samples'] = 64
    path.write_text(json.dumps(config, indent=2) + '\n')
    def cfg():
        return ConfigLoader(str(path), 'imptc', True, True, path.stem, 'stable_mdn', 'training')
    prefix = 'smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    a = cfg()
    os.environ['MDN_RUN_ID'] = prefix + '_continuous'
    assert training(a, '0')['status'] == 'completed'
    run_a = Path(a.result_path) / 'runs' / os.environ['MDN_RUN_ID']
    os.environ['MDN_RUN_ID'] = prefix + '_resume'
    os.environ['MDN_RESUME_CHECKPOINT'] = str(run_a / 'checkpoints/epoch_0001.pt')
    b = cfg()
    try:
        assert training(b, '0')['status'] == 'completed'
    finally:
        os.environ.pop('MDN_RESUME_CHECKPOINT', None)
    run_b = Path(b.result_path) / 'runs' / os.environ['MDN_RUN_ID']
    ca = torch.load(run_a / 'checkpoints/final.pt', map_location='cpu', weights_only=False)
    cb = torch.load(run_b / 'checkpoints/final.pt', map_location='cpu', weights_only=False)
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict'):
        assert equal(ca[key], cb[key]), key
    for key in ('train_nll', 'validation_nll', 'learning_rate'):
        assert [r[key] for r in ca['train_history']] == [r[key] for r in cb['train_history']]
    for run in (run_a, run_b):
        for name in ('best', 'last', 'final'):
            assert (run / f'checkpoints/{name}.pt').is_file()
        fixed = np.load(run / 'fixed_samples/inputs.npz')
        assert len(fixed['sample_ids']) == 8
        for prediction in (run / 'fixed_samples/predictions').glob('*.npz'):
            p = np.load(prediction)
            np.testing.assert_array_equal(p['sample_ids'], fixed['sample_ids'])
            for key in ('raw_output', 'pi', 'mu', 'sigma', 'rho', 'covariance'):
                assert np.isfinite(p[key]).all()
            np.testing.assert_allclose(p['pi'].sum(-1), 1., atol=1e-6)
            assert p['sigma'].min() >= .01 and np.abs(p['rho']).max() < 1
        rows = list(csv.DictReader((run / 'history.csv').open()))
        assert all(int(r['validation_sample_count']) == 19148 for r in rows)
        for metric in (run / 'metrics').glob('epoch_*.json'):
            m = json.loads(metric.read_text())
            assert m['validation_selection'] == 'full'
            assert m['metrics']['sample_count'] == 19148
            assert m['evaluator_version'] == 'stable_v2_ecdf_rng'
    loader = DataLoader(a)
    loader.load_eval_data()
    model = LSTM_Trajectory_Forecast(a.model_params)
    model.load_state_dict(ca['model_state_dict'])
    model.eval()
    total = 0.
    with torch.no_grad():
        for start in range(0, len(loader.eval_data[0]), 128):
            x = torch.as_tensor(loader.eval_data[0][start:start+128], dtype=torch.float32)
            y = torch.as_tensor(loader.eval_data[1][start:start+128], dtype=torch.float32)
            loss = -build_mdn_distribution(model(x), 3, a.model_params['mdn_parameterization']).log_prob(y).mean()
            total += float(loss) * len(x)
    nll = total / len(loader.eval_data[0])
    assert abs(nll - ca['train_history'][-1]['validation_nll']) < 1e-4
    manifest = json.loads((run_a / 'fixed_samples/manifest.json').read_text())
    indices = [loader.sample_keys['eval'].index(s['pickle_key']) for s in manifest['samples']]
    loader.eval_data = [data[indices] for data in loader.eval_data]
    loader.sample_ids['eval'] = [s['sample_id'] for s in manifest['samples']]
    a.eval_metrics.update(sharpness=True, asaee=True)
    a.test_params['mesh_resolution'] = 1.
    a.evaluation_path = str(run_a / 'smoke_fixed_metrics')
    Path(a.evaluation_path).mkdir()
    metrics = MDN_Forecaster(a, model, loader, 'eval', logging.getLogger('smoke'), torch.device('cpu')).evaluate(epoch=3)
    for key in ('ravg_percent', 'rmin_percent', 's68_m2_per_s', 's95_m2_per_s', 'asaee_m_per_s', 'minade20_m', 'minfde20_m'):
        assert np.isfinite(metrics[key])
    assert hashes() == before
    report = {'status': 'PASS', 'epochs': 3, 'train_samples_per_epoch': 189,
              'validation_samples': 19148, 'resume_from_epoch': 1,
              'model_optimizer_scheduler_equal': True, 'nll_history_equal': True,
              'validation_nll_recomputed': nll, 'baseline_unchanged': True,
              'runs': [str(run_a), str(run_b)], 'fixed_8_smoke_metrics': metrics,
              'limitations': ['Three epochs do not establish long-run stability or quality.',
                             'Smoke batch 128, reduced train subset, MC 64; fixed sharpness mesh 1 m.',
                             'Resume forks epoch 1; resume CSV has epochs 2-3, checkpoint history 1-3.']}
    output = ROOT / 'stable_mdn/reports/SMOKE_TRAINING.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('SMOKE PASS:', output)


if __name__ == '__main__':
    main()
