"""Head-growth correctness: decoded parameters, not just shapes."""
import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from growing_mdn.model import (BLOCKS, build_model, fc_row_map, grow_model,
                               new_component_rows, num_components)
from utils.mdn_distribution import build_mdn_distribution, decode_mdn_output

PARAMS = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'lstm_num_layers': 1,
          'num_gaussians': 3, 'output_factor': 6, 'forecast_horizon': 48,
          'mdn_parameterization': {'mode': 'legacy'}}


class GrowTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)

    def test_row_map_layout(self):
        rows = fc_row_map(3, 1, 48).reshape(48, BLOCKS, 4)
        for t in (0, 7, 47):
            for b in range(BLOCKS):
                for k in range(3):
                    self.assertEqual(int(rows[t, b, k]), t * 18 + b * 3 + k)
                self.assertEqual(int(rows[t, b, 3]), t * 18 + b * 3 + 1)

    def test_new_component_rows(self):
        rows = new_component_rows(3, 48)
        self.assertEqual(len(rows), 48 * BLOCKS)
        self.assertEqual(int(rows[0]), 3)
        self.assertEqual(int(rows[1]), 1 * 4 + 3)
        self.assertEqual(int(rows[BLOCKS]), 1 * 6 * 4 + 3)

    def test_grown_parameters(self):
        old = build_model(PARAMS, 3).eval()
        with torch.no_grad():
            old.fc.bias.normal_(0, 0.3)
        new, _ = grow_model(old, source=1, offsets=0.05)
        new.eval()
        self.assertEqual(num_components(new), 4)
        x = torch.randn(4, 32, 4)
        po, pn = decode_mdn_output(old(x), 3), decode_mdn_output(new(x), 4)
        for k in (0, 2):
            for key in ('pi', 'mu', 'sigma', 'rho'):
                torch.testing.assert_close(pn[key][:, :, k], po[key][:, :, k])
        for key in ('sigma', 'rho'):
            torch.testing.assert_close(pn[key][:, :, 1], po[key][:, :, 1])
            torch.testing.assert_close(pn[key][:, :, 3], po[key][:, :, 1])
        torch.testing.assert_close(pn['pi'][:, :, 1], pn['pi'][:, :, 3])
        torch.testing.assert_close(pn['pi'][:, :, 1] + pn['pi'][:, :, 3], po['pi'][:, :, 1])
        torch.testing.assert_close(pn['mu'][:, :, 1], po['mu'][:, :, 1] + 0.05)
        torch.testing.assert_close(pn['mu'][:, :, 3], po['mu'][:, :, 1] - 0.05)
        torch.testing.assert_close(pn['pi'].sum(-1), torch.ones(4, 48))

    def test_array_offsets_per_step(self):
        old = build_model(PARAMS, 3).eval()
        offsets = torch.rand(48, 2) * 0.1 + 0.01
        new, _ = grow_model(old, source=1, offsets=offsets)
        x = torch.randn(4, 32, 4)
        po, pn = decode_mdn_output(old(x.clone()), 3), decode_mdn_output(new.eval()(x), 4)
        torch.testing.assert_close(pn['mu'][:, :, 1], po['mu'][:, :, 1] + offsets, atol=1e-6, rtol=0)
        torch.testing.assert_close(pn['mu'][:, :, 3], po['mu'][:, :, 1] - offsets, atol=1e-6, rtol=0)

    def test_lstm_copied(self):
        old = build_model(PARAMS, 3)
        new, _ = grow_model(old, 0, 0.05)
        for (n1, p1), (n2, p2) in zip(old.lstm.named_parameters(), new.lstm.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2))

    def test_nll_continuity_after_split(self):
        old = build_model(PARAMS, 3).eval()
        x = torch.randn(2048, 32, 4)
        with torch.no_grad():
            dist_old = build_mdn_distribution(old(x), 3)
            y = dist_old.sample()
            nll_old = -dist_old.log_prob(y).mean()
            new, _ = grow_model(old, 2, 0.05)
            nll_new = -build_mdn_distribution(new.eval()(x), 4).log_prob(y).mean()
        self.assertLess(abs(float(nll_new - nll_old)), 1e-3)

    def test_invalid_arguments(self):
        old = build_model(PARAMS, 3)
        with self.assertRaises(ValueError):
            grow_model(old, 3, 0.05)
        with self.assertRaises(ValueError):
            grow_model(old, -1, 0.05)
        with self.assertRaises(ValueError):
            grow_model(old, 0, 0.0)
        with self.assertRaises(ValueError):
            grow_model(old, 0, torch.full((48, 2), float('nan')))
        with self.assertRaises(ValueError):
            grow_model(old, 0, torch.ones(47, 2))
        with self.assertRaises(ValueError):
            grow_model(old, 0, -torch.ones(48, 2))


if __name__ == '__main__':
    unittest.main()
