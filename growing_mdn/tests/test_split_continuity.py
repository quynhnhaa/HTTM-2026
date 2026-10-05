"""Regression: a split must be continuous on a model with very small sigma."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from growing_mdn.model import build_model, grow_model
from growing_mdn.selection import source_sigma_median
from growing_mdn.train import check_split_continuity
from utils.mdn_distribution import build_mdn_distribution

PARAMS = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'lstm_num_layers': 1,
          'num_gaussians': 1, 'output_factor': 6, 'forecast_horizon': 48,
          'mdn_parameterization': {'mode': 'legacy'}}
T = 48


def tiny_sigma_model():
    model = build_model(PARAMS, 1).eval()
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        bias = model.fc.bias.view(T, 6, 1)
        log_sigma = torch.log(0.001 * (1 + torch.arange(T, dtype=torch.float32)))
        bias[:, 2, 0] = log_sigma
        bias[:, 3, 0] = log_sigma
    return model


def mean_nll(model, x, y, k):
    with torch.no_grad():
        return float(-build_mdn_distribution(model.eval()(x), k).log_prob(y).mean())


class SplitContinuityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)
        self.model = tiny_sigma_model()
        self.x = torch.randn(512, 32, 4)
        with torch.no_grad():
            self.y = build_mdn_distribution(self.model(self.x), 1).sample()

    def test_source_sigma_median_matches_bias(self):
        med = source_sigma_median(self.model, self.x.numpy(), 'cpu', 100, 0)
        expected = 0.001 * (1 + np.arange(T))
        self.assertEqual(med.shape, (T, 2))
        np.testing.assert_allclose(med, np.stack([expected, expected], 1), rtol=1e-5)

    def test_absolute_split_is_discontinuous_relative_is_not(self):
        before = mean_nll(self.model, self.x, self.y, 1)
        absolute, _ = grow_model(self.model, 0, 0.05)
        self.assertGreater(abs(mean_nll(absolute, self.x, self.y, 2) - before), 10)
        offsets = 0.05 * source_sigma_median(self.model, self.x.numpy(), 'cpu', 100, 0)
        relative, _ = grow_model(self.model, 0, offsets)
        self.assertLess(abs(mean_nll(relative, self.x, self.y, 2) - before), 0.01)

    def test_check_split_continuity(self):
        check_split_continuity(1.0, 1.005, 0.01)
        for before, after in ((1.0, 1.02), (1.0, float('nan')), (1.0, float('inf')), (float('nan'), 1.0)):
            with self.subTest(before=before, after=after):
                with self.assertRaises(RuntimeError):
                    check_split_continuity(before, after, 0.01)


if __name__ == '__main__':
    unittest.main()
