import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / 'base_mdn')):
    if p not in sys.path:
        sys.path.insert(0, p)
from base_lstm import LSTM_Trajectory_Forecast, NLL_MDN_loss  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402
from sparsemax_mdn import model as model_module  # noqa: E402
from sparsemax_mdn.model import SparsemaxMDN, SENTINEL  # noqa: E402
from sparsemax_mdn.loss import sparsemax_nll  # noqa: E402
from sparsemax_mdn.sparsemax import sparsemax  # noqa: E402

K = 8
CFG = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'num_gaussians': K, 'output_factor': 6,
       'forecast_horizon': 48, 'lstm_num_layers': 1, 'mdn_parameterization': {'mode': 'legacy'}}


def make_model(scale=1.0, seed=0):
    torch.manual_seed(seed)
    m = SparsemaxMDN(CFG)
    with torch.no_grad():
        m.fc.bias.add_(torch.randn_like(m.fc.bias) * scale)
    return m


class ModelTest(unittest.TestCase):
    def test_shapes_and_decode_equals_sparsemax(self):
        m = make_model(scale=2.0)
        x = torch.randn(5, 32, 4)
        raw = m(x)
        self.assertEqual(tuple(raw.shape), (5, 48, 6 * K))
        base = LSTM_Trajectory_Forecast(CFG)
        base.load_state_dict(m.state_dict())
        plain = base(x)
        torch.testing.assert_close(raw[..., :5 * K], plain[..., :5 * K], atol=0, rtol=0)
        pi = sparsemax(plain[..., 5 * K:])
        decoded = decode_mdn_output(raw, K)['pi']
        self.assertTrue(((pi == 0) == (decoded == 0)).all())
        self.assertTrue((pi == 0).any(), 'test needs some zero weights')
        torch.testing.assert_close(decoded, pi, atol=1e-6, rtol=0)
        self.assertTrue((raw[..., 5 * K:][pi == 0] == SENTINEL).all())

    def test_rejects_non_legacy_parameterization(self):
        bad = dict(CFG, mdn_parameterization={'mode': 'other'})
        with self.assertRaises(ValueError):
            SparsemaxMDN(bad)

    def test_gradient_through_model_is_finite(self):
        m = make_model(scale=2.0)
        raw = m(torch.randn(6, 32, 4))
        loss, div = sparsemax_nll(raw, torch.randn(6, 48, 2), K)
        self.assertFalse(div)
        loss.backward()
        for p in m.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())


def reference_nll(pi, mu, sigma, rho, y):
    total = []
    for b in range(pi.shape[0]):
        for t in range(pi.shape[1]):
            dens = 0.0
            for k in range(pi.shape[2]):
                if pi[b, t, k] == 0:
                    continue
                sx, sy, r = sigma[b, t, k, 0], sigma[b, t, k, 1], rho[b, t, k]
                dx = (y[b, t, 0] - mu[b, t, k, 0]) / sx
                dy = (y[b, t, 1] - mu[b, t, k, 1]) / sy
                z = (dx * dx - 2 * r * dx * dy + dy * dy) / (1 - r * r)
                dens += pi[b, t, k] * math.exp(-z / 2) / (2 * math.pi * sx * sy * math.sqrt(1 - r * r))
            total.append(-math.log(dens))
    return float(np.mean(total))


def random_raw(batch=4, steps=6, k=K, seed=0, with_zeros=True):
    g = torch.Generator().manual_seed(seed)
    blocks = [torch.randn(batch, steps, k, generator=g, dtype=torch.float64) for _ in range(5)]
    logit = torch.randn(batch, steps, k, generator=g, dtype=torch.float64) * (3 if with_zeros else 0.1)
    pi = sparsemax(logit)
    log_pi = torch.where(pi > 0, torch.log(pi.clamp_min(1e-300)), torch.full_like(pi, SENTINEL))
    return torch.cat(blocks[:5] + [log_pi], -1), pi


class LossTest(unittest.TestCase):
    def test_matches_independent_reference_with_zero_weights(self):
        raw, pi = random_raw()
        self.assertTrue((pi == 0).any())
        y = torch.randn(4, 6, 2, dtype=torch.float64)
        loss, div = sparsemax_nll(raw, y, K)
        self.assertFalse(div)
        d = decode_mdn_output(raw, K)
        ref = reference_nll(pi.numpy(), d['mu'].numpy(), d['sigma'].numpy(), d['rho'].numpy(), y.numpy())
        self.assertAlmostEqual(loss.item(), ref, places=8)

    def test_zero_weight_component_gets_exactly_zero_gradient(self):
        raw, pi = random_raw(seed=3)
        raw = raw.clone().requires_grad_(True)
        y = torch.randn(4, 6, 2, dtype=torch.float64)
        loss, _ = sparsemax_nll(raw, y, K)
        loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        zero = pi == 0
        self.assertTrue(zero.any())
        for block in (0, 1, 2, 3, 4):  # mu_x, mu_y, log sx, log sy, rho_pre
            g = raw.grad[..., block * K:(block + 1) * K]
            self.assertTrue((g[zero] == 0).all(), f'block {block}')
        # active components do receive gradient
        self.assertTrue((raw.grad[..., :K][~zero] != 0).any())

    def test_end_to_end_gradient_wrt_zero_weight_component_params(self):
        m = make_model(scale=2.5, seed=1)
        x = torch.randn(8, 32, 4)
        raw = m(x).detach().double()
        raw[..., 4 * K:5 * K] = raw[..., 4 * K:5 * K].clamp(-2, 2)  # keep covariances PD
        raw.requires_grad_(True)
        pi = decode_mdn_output(raw, K)['pi']
        loss, _ = sparsemax_nll(raw, torch.randn(8, 48, 2, dtype=torch.float64), K)
        loss.backward()
        zero = pi == 0
        self.assertTrue(zero.any())
        self.assertTrue((raw.grad[..., :K][zero] == 0).all())
        self.assertTrue((raw.grad[..., 2 * K:3 * K][zero] == 0).all())

    def test_non_finite_returns_diverged(self):
        raw, _ = random_raw()
        raw = raw.clone()
        raw[0, 0, 0] = float('nan')
        loss, div = sparsemax_nll(raw, torch.zeros(4, 6, 2, dtype=torch.float64), K)
        self.assertTrue(div)
        self.assertIsNone(loss)

    def test_equals_baseline_nll_when_sparsemax_is_softmax(self):
        """Guards against an accidental change of the NLL definition."""
        with mock.patch.object(model_module, 'sparsemax', lambda z, dim=-1: torch.softmax(z, dim=dim)):
            m = make_model(scale=0.3, seed=2)
            x = torch.randn(7, 32, 4)
            raw = m(x)
        y = torch.randn(7, 48, 2) * 2
        mine, d1 = sparsemax_nll(raw, y, K)
        base, d2 = NLL_MDN_loss(raw, y, K, {'mode': 'legacy'})
        self.assertFalse(d1 or d2)
        self.assertLess(abs(mine.item() - base.item()), 1e-5)


if __name__ == '__main__':
    unittest.main()
