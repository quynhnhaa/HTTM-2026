import copy
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / 'base_mdn')):
    if p not in sys.path:
        sys.path.insert(0, p)
from base_lstm import LSTM_Trajectory_Forecast, NLL_MDN_loss  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402
from cvres_mdn import base_hashes  # noqa: E402
from cvres_mdn.model import CVResidualMDN, cv_extrapolation  # noqa: E402

K, T = 3, 48
CFG = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'num_gaussians': K, 'output_factor': 6, 'forecast_horizon': T,
       'lstm_num_layers': 1, 'delta_t': 0.1, 'cv_residual': {'velocity_columns': [2, 3]}}
CONF = ROOT / 'cvres_mdn/configs/imptc'
BASE = ROOT / 'base_mdn/configs/imptc'


class ExtrapolationTest(unittest.TestCase):
    def test_values(self):
        x = torch.zeros(2, 32, 4)
        x[0, -1, 2:4] = torch.tensor([1.0, 0.0])
        x[1, -1, 2:4] = torch.tensor([0.0, -2.0])
        out = cv_extrapolation(x, (2, 3), 0.1, T)
        self.assertEqual(out.shape, (2, T, 2))
        torch.testing.assert_close(out[0, :, 0], 0.1 * torch.arange(1, T + 1, dtype=torch.float32))
        torch.testing.assert_close(out[1, :, 1], -0.2 * torch.arange(1, T + 1, dtype=torch.float32))
        self.assertTrue(bool((out[0, :, 1] == 0).all()))

    def test_only_the_last_observed_step_matters(self):
        x = torch.randn(3, 32, 4)
        y = x.clone()
        y[:, :-1] += 5.0
        torch.testing.assert_close(cv_extrapolation(x, (2, 3), 0.1, T), cv_extrapolation(y, (2, 3), 0.1, T))


class ModelTest(unittest.TestCase):
    def test_zero_head_gives_exactly_the_cv_line_for_every_component(self):
        m = CVResidualMDN(CFG)
        with torch.no_grad():
            m.fc.weight.zero_()
            m.fc.bias.zero_()
        x = torch.randn(4, 32, 4)
        d = decode_mdn_output(m(x), K)
        cv = cv_extrapolation(x, (2, 3), 0.1, T)
        for i in range(K):
            torch.testing.assert_close(d['mu'][:, :, i, :], cv)
        torch.testing.assert_close(d['sigma'], torch.ones_like(d['sigma']))
        torch.testing.assert_close(d['pi'], torch.full_like(d['pi'], 1 / K))

    def test_only_mu_blocks_differ_from_the_plain_baseline(self):
        torch.manual_seed(0)
        plain = LSTM_Trajectory_Forecast(CFG)
        m = CVResidualMDN(CFG)
        m.load_state_dict(plain.state_dict())
        x = torch.randn(3, 32, 4)
        a, b = plain(x), m(x)
        torch.testing.assert_close(a[..., 2 * K:], b[..., 2 * K:])        # sigma, rho, pi blocks untouched
        cv = cv_extrapolation(x, (2, 3), 0.1, T)
        torch.testing.assert_close(b[..., :K] - a[..., :K], cv[..., 0:1].expand(-1, -1, K))
        torch.testing.assert_close(b[..., K:2 * K] - a[..., K:2 * K], cv[..., 1:2].expand(-1, -1, K))

    def test_same_state_dict_keys_and_parameter_count_as_baseline(self):
        plain, m = LSTM_Trajectory_Forecast(CFG), CVResidualMDN(CFG)
        self.assertEqual(list(plain.state_dict().keys()), list(m.state_dict().keys()))
        self.assertEqual(sum(p.numel() for p in plain.parameters()), sum(p.numel() for p in m.parameters()))

    def test_baseline_loss_is_finite_and_gradients_flow(self):
        torch.manual_seed(1)
        m = CVResidualMDN(CFG)
        loss, diverged = NLL_MDN_loss(m(torch.randn(8, 32, 4)), torch.randn(8, T, 2), K, None)
        self.assertFalse(diverged)
        loss.backward()
        for name, p in m.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_rejects_invalid_velocity_columns(self):
        for bad in ([2], [2, 4], [1, 2, 3]):
            cfg = copy.deepcopy(CFG)
            cfg['cv_residual']['velocity_columns'] = bad
            with self.assertRaises(ValueError):
                CVResidualMDN(cfg)


class DataFactsTest(unittest.TestCase):
    def test_recorded_conventions_hold(self):
        facts = json.loads((ROOT / 'cvres_mdn/reports/CV_PRIOR_CHECK.json').read_text())['facts']
        self.assertLess(facts['last_obs_position_abs_max'], 1e-3)
        self.assertGreater(min(facts['corr_vel_vs_diff_pos']), 0.9999)
        self.assertAlmostEqual(facts['magnitude_ratio_diff_over_vel'], 0.1, places=3)
        self.assertGreater(min(facts['corr_y0_vs_last_displacement']), 0.99)


class ConfigTest(unittest.TestCase):
    def test_full_config_differs_from_baseline_only_in_declared_keys(self):
        full = json.loads((CONF / 'cvres_peds_imptc.json').read_text())
        base = json.loads((BASE / 'default_peds_imptc.json').read_text())
        self.assertEqual(full['model_params'].pop('cv_residual'), {'velocity_columns': [2, 3]})
        for key in ('experiment_name', 'method'):
            full['experiment_params'].pop(key)
            base['experiment_params'].pop(key, None)
        self.assertEqual(full, base)

    def test_smoke_config_is_small(self):
        self.assertLessEqual(json.loads((CONF / 'smoke_cvres_peds_imptc.json').read_text())['train_params']['train_epochs'], 3)


class HashGuardTest(unittest.TestCase):
    def test_base_unchanged(self):
        base_hashes.verify()


if __name__ == '__main__':
    unittest.main()
