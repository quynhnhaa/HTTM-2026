"""Review completed residual smoke artifacts; no training."""
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'residual_mdn'))
sys.path.insert(0, str(ROOT / 'stable_mdn'))
from model import ResidualMDN
from eval import MDN_Forecaster
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader


def main():
    run = ROOT / 'results/trained_models/residual_mdn/imptc/smoke_residual_peds_imptc/runs/residual_smoke_seed2024'
    continuation = run.parent / 'residual_continuation_smoke'
    assert json.loads((run / 'run_manifest.json').read_text())['status'] == 'completed'
    assert json.loads((continuation / 'run_manifest.json').read_text())['status'] == 'completed'
    for name in ('best', 'last', 'final', 'epoch_0001', 'epoch_0002', 'epoch_0003'):
        assert (run / 'checkpoints' / f'{name}.pt').is_file()
    inputs = np.load(run / 'fixed_samples/inputs.npz')
    assert len(inputs['sample_ids']) == 8
    for capture in (run / 'fixed_samples/predictions').glob('*.npz'):
        p = np.load(capture)
        np.testing.assert_array_equal(p['sample_ids'], inputs['sample_ids'])
        for key in ('raw_output', 'pi', 'mu', 'sigma', 'rho', 'covariance', 'cv_trajectory', 'residual_mu'):
            assert np.isfinite(p[key]).all()
        np.testing.assert_allclose(p['mu'], p['residual_mu'] + p['cv_trajectory'][:, :, None, :], atol=1e-6)
        np.testing.assert_allclose(p['pi'].sum(-1), 1., atol=1e-6)
        assert p['sigma'].min() >= .01 and np.abs(p['rho']).max() < 1
    rows = list(csv.DictReader((run / 'history.csv').open()))
    assert len(rows) == 3 and all(int(r['validation_sample_count']) == 19148 for r in rows)
    parent_ck = torch.load(run / 'checkpoints/final.pt', map_location='cpu', weights_only=False)
    next_ck = torch.load(continuation / 'checkpoints/final.pt', map_location='cpu', weights_only=False)
    assert next_ck['epoch'] == 5 and next_ck['scheduler_state_dict']['last_epoch'] == 2
    for k, state in next_ck['optimizer_state_dict']['state'].items():
        assert int(state['step']) == int(parent_ck['optimizer_state_dict']['state'][k]['step']) + 4
    full = json.loads((ROOT / 'residual_mdn/configs/imptc/residual_peds_imptc.json').read_text())
    baseline = json.loads((ROOT / 'stable_mdn/configs/imptc/stable_peds_imptc.json').read_text())
    for key in ('train_params', 'test_params', 'eval_metrics'):
        assert full[key] == baseline[key]
    assert {k: v for k, v in full['model_params'].items() if k != 'residual_velocity_steps'} == baseline['model_params']
    cfg = ConfigLoader(str(ROOT / 'residual_mdn/configs/imptc/smoke_residual_peds_imptc.json'),
                       'imptc', False, False, 'smoke_residual_peds_imptc', 'residual_mdn', 'training')
    loader = DataLoader(cfg)
    loader.load_eval_data()
    manifest = json.loads((run / 'fixed_samples/manifest.json').read_text())
    indices = [loader.sample_keys['eval'].index(s['pickle_key']) for s in manifest['samples']]
    loader.eval_data = [a[indices] for a in loader.eval_data]
    loader.sample_ids['eval'] = inputs['sample_ids'].tolist()
    model = ResidualMDN(cfg.model_params)
    best = torch.load(run / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    cfg.eval_metrics.update(sharpness=True, asaee=True)
    cfg.test_params['mesh_resolution'] = 1.
    cfg.evaluation_path = str(run / 'fixed_metric_smoke')
    Path(cfg.evaluation_path).mkdir(exist_ok=True)
    metrics = MDN_Forecaster(cfg, model, loader, 'eval', logging.getLogger('smoke'), torch.device('cpu')).evaluate(3)
    for key in ('ravg_percent', 'rmin_percent', 's68_m2_per_s', 's95_m2_per_s', 'minade20_m', 'minfde20_m', 'asaee_m_per_s'):
        assert np.isfinite(metrics[key])
    reports = ROOT / 'residual_mdn/reports'
    reports.mkdir(exist_ok=True)
    capture = np.load(run / 'fixed_samples/predictions/best.npz')
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        observed, target = loader.eval_data[0][i], loader.eval_data[1][i]
        ax.plot(observed[:, 0], observed[:, 1], color='#0072B2', label='Observed')
        ax.plot(target[:, 0], target[:, 1], color='black', label='Ground truth')
        cv = capture['cv_trajectory'][i]
        ax.plot(cv[:, 0], cv[:, 1], '--', color='#D55E00', label='CV prior')
        for k in range(3):
            mu = capture['mu'][i, :, k]
            ax.plot(mu[:, 0], mu[:, 1], ':', alpha=.8, label=f'Component mean {k+1}')
        ax.set_title(f'Fixed validation sample {i+1}')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('ego x (m)'); ax.set_ylabel('ego y (m)'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Residual MDN smoke (3 epochs): CV prior and corrected component means\nComponent indices are per-timestep; these lines do not imply joint trajectory modes.')
    fig.tight_layout(); fig.savefig(reports / 'SMOKE_CV_COMPONENT_MEANS.png', dpi=130); plt.close(fig)
    report = {'status': 'PASS', 'epochs': 3, 'validation_samples': 19148,
              'fixed_samples': 8, 'parameter_count': 8224,
              'cv_plus_residual_equals_final_mu': True,
              'continuation_smoke': {'epochs': 2, 'adam_step_increment': 4},
              'full_protocol_matches_stable_baseline': True,
              'fixed_8_metric_smoke': metrics,
              'limitations': ['Three epochs do not establish prediction quality or long-run stability.',
                             'Smoke: 189 train samples/epoch, batch128, MC64; fixed metric mesh1m.']}
    (reports / 'SMOKE.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Residual smoke review PASS:', reports / 'SMOKE.json')


if __name__ == '__main__':
    main()
