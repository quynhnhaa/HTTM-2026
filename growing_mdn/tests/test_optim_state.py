import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from growing_mdn.model import build_model, grow_model, new_component_rows
from growing_mdn.optim_state import remap_adam_state
from utils.mdn_distribution import build_mdn_distribution

PARAMS = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'lstm_num_layers': 1,
          'num_gaussians': 3, 'output_factor': 6, 'forecast_horizon': 48,
          'mdn_parameterization': {'mode': 'legacy'}}


def one_step(model, optimizer, k):
    x, y = torch.randn(16, 32, 4), torch.randn(16, 48, 2)
    optimizer.zero_grad()
    loss = -build_mdn_distribution(model(x), k).log_prob(y).mean()
    loss.backward()
    optimizer.step()
    return float(loss)


class RemapTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)
        self.old = build_model(PARAMS, 3)
        self.old_opt = torch.optim.Adam(self.old.parameters(), lr=1e-3)
        one_step(self.old, self.old_opt, 3)
        one_step(self.old, self.old_opt, 3)
        self.new, self.rows = grow_model(self.old, 1, 0.05)
        self.new_opt = torch.optim.Adam(self.new.parameters(), lr=1e-3)

    def test_fc_moments_gathered_and_lstm_copied(self):
        remap_adam_state(self.old_opt, self.new_opt, self.old, self.new, self.rows)
        for name in ('weight', 'bias'):
            old_state = self.old_opt.state[getattr(self.old.fc, name)]
            new_state = self.new_opt.state[getattr(self.new.fc, name)]
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(new_state[key], old_state[key][self.rows])
            self.assertEqual(float(new_state['step']), float(old_state['step']))
        for (_, po), (_, pn) in zip(self.old.lstm.named_parameters(), self.new.lstm.named_parameters()):
            for key in ('exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(self.old_opt.state[po][key], self.new_opt.state[pn][key]))

    def test_zero_new_rows(self):
        remap_adam_state(self.old_opt, self.new_opt, self.old, self.new, self.rows, zero_new_rows=True)
        rows = new_component_rows(3, 48)
        state = self.new_opt.state[self.new.fc.weight]
        self.assertEqual(float(state['exp_avg'][rows].abs().sum()), 0.0)
        self.assertEqual(float(state['exp_avg_sq'][rows].abs().sum()), 0.0)
        kept = torch.ones(state['exp_avg'].shape[0], dtype=torch.bool)
        kept[rows] = False
        self.assertGreater(float(state['exp_avg'][kept].abs().sum()), 0.0)

    def test_training_continues_finite(self):
        remap_adam_state(self.old_opt, self.new_opt, self.old, self.new, self.rows)
        for _ in range(3):
            self.assertTrue(torch.isfinite(torch.tensor(one_step(self.new, self.new_opt, 4))))


if __name__ == '__main__':
    unittest.main()
