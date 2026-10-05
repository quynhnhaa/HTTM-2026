"""Verify the physical prior, MDN layout and gradients without training."""
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'residual_mdn'))
sys.path.insert(0, str(ROOT / 'stable_mdn'))
from model import ResidualMDN
from base_lstm import LSTM_Trajectory_Forecast, NLL_MDN_loss
from utils.mdn_distribution import decode_mdn_output


class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'residual_mdn/configs/imptc/residual_peds_imptc.json').read_text())['model_params']
        torch.manual_seed(2024)
        self.model = ResidualMDN(self.cfg)

    def test_zero_residual_is_cv_and_correct_time(self):
        with torch.no_grad():
            self.model.fc.weight.zero_()
            self.model.fc.bias.zero_()
        x = torch.zeros(2, 32, 4)
        x[:, -1, :2] = torch.tensor([10., -3.])
        x[:, -5:, 2:4] = torch.tensor([2., -1.])
        # Earlier velocities must not affect the prior.
        x[:, :-5, 2:4] = 100.
        out = decode_mdn_output(self.model(x), 3, self.cfg['mdn_parameterization'])
        expected = torch.tensor([10., -3.])[None, None, :] + torch.tensor([2., -1.])[None, None, :] * torch.arange(1, 49)[None, :, None] * .1
        torch.testing.assert_close(out['mu'], expected[:, :, None, :].expand(2, 48, 3, 2))
        torch.testing.assert_close(out['mu'][0, 0, 0], torch.tensor([10.2, -3.1]))
        torch.testing.assert_close(out['sigma'], torch.full((2, 48, 3, 2), 1.01))

    def test_only_means_change_and_parameter_count_is_identical(self):
        x = torch.randn(2, 32, 4)
        raw = self.model.residual_output(x)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 48, 18))
        torch.testing.assert_close(output[..., 6:], raw[..., 6:], rtol=0, atol=0)
        for key in ('pi', 'sigma', 'rho', 'covariance'):
            a = decode_mdn_output(raw, 3, self.cfg['mdn_parameterization'])[key]
            b = decode_mdn_output(output, 3, self.cfg['mdn_parameterization'])[key]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 8224)
        baseline = LSTM_Trajectory_Forecast(self.cfg)
        self.assertEqual(set(self.model.state_dict()), set(baseline.state_dict()))
        torch.manual_seed(17)
        residual = ResidualMDN(self.cfg)
        torch.manual_seed(17)
        baseline = LSTM_Trajectory_Forecast(self.cfg)
        for key, value in residual.state_dict().items():
            torch.testing.assert_close(value, baseline.state_dict()[key], rtol=0, atol=0)

    def test_position_nll_has_finite_gradients(self):
        output = self.model(torch.randn(3, 32, 4))
        target = torch.randn(3, 48, 2)
        loss, failed = NLL_MDN_loss(output, target, 3, self.cfg['mdn_parameterization'])
        self.assertFalse(failed)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))


if __name__ == '__main__':
    unittest.main()
