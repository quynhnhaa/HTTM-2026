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
from base_lstm import NLL_MDN_loss  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402
from kappa_mdn import base_hashes  # noqa: E402
from kappa_mdn.artifacts import tau_at  # noqa: E402
from kappa_mdn.loss import LossStats, kappa_nll, make_loss_fn  # noqa: E402
from kappa_mdn.model import KappaMDN, hard_k, hard_open, log_gate  # noqa: E402

K = 16
T = 48
CFG = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'num_gaussians': K, 'output_factor': 6,
       'forecast_horizon': T, 'lstm_num_layers': 1, 'mdn_parameterization': {'mode': 'legacy'},
       'kappa': {'init': 12.0, 'tau_start': 1.0}}
CONF = ROOT / 'kappa_mdn/configs/imptc'
BASE = ROOT / 'base_mdn/configs/imptc'


def make_model(kappa=None, tau=1.0, seed=0):
    torch.manual_seed(seed)
    m = KappaMDN(CFG)
    with torch.no_grad():
        if kappa is not None:
            m.set_kappa(kappa)
        m.tau.fill_(tau)
    return m


class GateMathTest(unittest.TestCase):
    def test_log_gate_matches_sigmoid_and_is_monotone(self):
        kappa = torch.full((T,), 3.0)
        g = log_gate(kappa, 0.5, K).exp()
        j = torch.arange(1, K + 1, dtype=torch.float32)
        torch.testing.assert_close(g[0], torch.sigmoid((3.0 - j + 0.5) / 0.5))
        self.assertTrue(bool((g[0, :-1] >= g[0, 1:]).all()))
        self.assertGreater(float(g[0, 2]), 0.5)   # j = 3 <= kappa
        self.assertLess(float(g[0, 3]), 0.5)      # j = 4 > kappa

    def test_log_gate_is_finite_far_from_kappa(self):
        lg = log_gate(torch.full((T,), 1.0), 0.1, K)
        self.assertTrue(torch.isfinite(lg).all())

    def test_hard_k_rounds_and_clamps(self):
        k = torch.tensor([0.2, 1.4, 2.6, 16.9, 40.0])
        self.assertEqual(hard_k(k, K).tolist(), [1.0, 1.0, 3.0, 16.0, 16.0])
        self.assertEqual(hard_open(k, K).sum(-1).tolist(), [1, 1, 3, 16, 16])


class ModelTest(unittest.TestCase):
    def test_eval_gives_exact_zeros_above_k_and_normalised_weights(self):
        m = make_model(kappa=3.2).eval()
        x = torch.randn(5, 32, 4)
        with torch.no_grad():
            raw = m(x)
            pi = decode_mdn_output(raw, K)['pi']
        self.assertTrue(torch.equal(pi[..., 3:], torch.zeros_like(pi[..., 3:])))
        torch.testing.assert_close(pi.sum(-1), torch.ones(5, T))
        self.assertTrue(bool((pi[..., :3] > 0).all()))
        self.assertEqual(raw.shape, (5, T, 6 * K))

    def test_eval_open_weights_are_softmax_over_open_logits(self):
        m = make_model(kappa=4.0).eval()
        x = torch.randn(2, 32, 4)
        with torch.no_grad():
            raw = m(x)
            fc = torch.reshape(m.fc(m.lstm(x)[0][:, -1, :]), (-1, T, 6 * K))[..., 5 * K:]
            pi = decode_mdn_output(raw, K)['pi']
        torch.testing.assert_close(pi[..., :4], torch.softmax(fc[..., :4], -1))

    def test_train_mode_keeps_all_weights_positive_and_kappa_gets_gradient(self):
        m = make_model(kappa=8.0, tau=1.0).train()
        x, y = torch.randn(6, 32, 4), torch.randn(6, T, 2)
        raw = m(x)
        self.assertTrue(bool((decode_mdn_output(raw, K)['pi'] > 0).all()))
        loss, diverged = kappa_nll(raw, y, K, unnormalised_log_weights=True)
        self.assertFalse(diverged)
        loss.backward()
        self.assertIsNotNone(m.kappa_logit.grad)
        self.assertTrue(torch.isfinite(m.kappa_logit.grad).all())
        self.assertGreater(float(m.kappa_logit.grad.abs().sum()), 0.0)

    def test_nll_gradient_wants_more_components_when_a_closed_one_is_needed(self):
        m = make_model(kappa=3.0, tau=0.3).train()
        with torch.no_grad():
            m.fc.weight.zero_()
            m.fc.bias.zero_()
            b = m.fc.bias.view(T, 6, K)
            b[:, 2:4, :] = -1.0           # log sigma
            b[:, 0:2, 4] = 5.0            # component 5 sits at (5, 5); the others at (0, 0)
        raw = m(torch.randn(3, 32, 4))
        target = torch.full((3, T, 2), 5.0)
        loss, _ = kappa_nll(raw, target, K, unnormalised_log_weights=True)
        loss.backward()
        self.assertLess(float(m.kappa_logit.grad.sum()), 0.0)  # raising kappa lowers the NLL

    def test_penalty_gradient_is_positive_in_kappa(self):
        m = make_model(kappa=5.0, tau=0.5)
        m.soft_open_count().backward()
        self.assertTrue(bool((m.kappa_logit.grad > 0).all()))

    def test_state_dict_roundtrip_keeps_kappa_and_tau(self):
        a = make_model(kappa=6.5, tau=0.25, seed=1)
        b = KappaMDN(CFG)
        b.load_state_dict(a.state_dict())
        self.assertTrue(torch.equal(a.kappa, b.kappa))
        self.assertEqual(float(a.tau), float(b.tau))
        a.eval(); b.eval()
        x = torch.randn(2, 32, 4)
        torch.testing.assert_close(a(x), b(x))

    def test_kappa_stays_inside_one_and_kmax_for_any_logit(self):
        m = make_model()
        for value in (-1e4, -50.0, 0.0, 50.0, 1e4):
            m.kappa_logit.data.fill_(value)
            self.assertTrue(bool(((m.kappa >= 1.0) & (m.kappa <= K)).all()))

    def test_init_value_is_recovered(self):
        torch.testing.assert_close(make_model().kappa, torch.full((T,), 12.0))

    def test_closing_a_gate_removes_mass_and_cannot_be_compensated(self):
        # Training weights are softmax * gate without renormalisation: total mass equals the open mass, < 1.
        m = make_model(kappa=2.0, tau=0.05).train()
        with torch.no_grad():
            raw = m(torch.randn(3, 32, 4))
            mass = raw[..., 5 * K:].exp().sum(-1)
        self.assertTrue(bool((mass < 1.0).all()))
        self.assertTrue(bool((mass > 0.0).all()))

    def test_rejects_non_legacy_parameterization(self):
        cfg = copy.deepcopy(CFG)
        cfg['mdn_parameterization'] = {'mode': 'paper'}
        with self.assertRaises(ValueError):
            KappaMDN(cfg)


class LossTest(unittest.TestCase):
    def test_equals_baseline_nll_when_all_gates_open(self):
        m = make_model(kappa=16.0, tau=0.01).train()
        x, y = torch.randn(8, 32, 4), torch.randn(8, T, 2)
        with torch.no_grad():
            raw = m(x)
            ours, _ = kappa_nll(raw, y, K, unnormalised_log_weights=True)
            base, _ = NLL_MDN_loss(raw, y, K, None)
        self.assertLess(abs(float(ours) - float(base)), 1e-5)

    def test_penalty_only_in_training_and_stats_recorded(self):
        m = make_model(kappa=8.0, tau=1.0)
        stats = LossStats()
        fn = make_loss_fn(m, K, 0.01, stats)
        x, y = torch.randn(4, 32, 4), torch.randn(4, T, 2)
        m.train()
        raw = m(x)
        objective, _ = fn(raw, y)
        nll, _ = kappa_nll(raw, y, K, unnormalised_log_weights=True)
        expected = float(nll.detach()) + 0.01 * float(m.soft_open_count().detach())
        self.assertAlmostEqual(float(objective.detach()), expected, places=5)
        consumed = stats.consume()
        self.assertAlmostEqual(consumed['train_nll'], float(nll.detach()), places=5)
        self.assertIsNone(stats.consume())
        m.eval()
        with torch.no_grad():
            raw = m(x)
            pure, _ = fn(raw, y)
            ref, _ = kappa_nll(raw, y, K)
        self.assertEqual(float(pure), float(ref))
        self.assertIsNone(stats.consume())  # nothing recorded in eval mode

    def test_non_finite_is_reported_as_divergence(self):
        raw = torch.full((1, T, 6 * K), float('nan'))
        loss, diverged = kappa_nll(raw, torch.zeros(1, T, 2), K)
        self.assertTrue(diverged)
        self.assertIsNone(loss)


class ScheduleTest(unittest.TestCase):
    def test_tau_schedule(self):
        self.assertAlmostEqual(tau_at(1, 1.0, 0.1, 1250), 1.0)
        self.assertAlmostEqual(tau_at(1250, 1.0, 0.1, 1250), 0.1)
        self.assertAlmostEqual(tau_at(2500, 1.0, 0.1, 1250), 0.1)
        values = [tau_at(e, 1.0, 0.1, 1250) for e in range(1, 1300, 50)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertAlmostEqual(tau_at(5, 1.0, 0.1, 1), 0.1)


class ConfigTest(unittest.TestCase):
    def load(self, path):
        return json.loads(path.read_text())

    def test_full_config_differs_from_baseline_only_in_declared_keys(self):
        full, base = self.load(CONF / 'kappa_k16_peds_imptc.json'), self.load(BASE / 'default_peds_imptc.json')
        self.assertEqual(full['model_params']['num_gaussians'], 16)
        self.assertEqual(full['model_params']['kappa'], {'init': 12.0, 'tau_start': 1.0})
        self.assertEqual(full['experiment_params']['kappa'],
                         {'tau_end': 0.1, 'anneal_epochs': 1250, 'penalty_lambda': 0.01, 'lr_multiplier': 10.0})
        a, b = copy.deepcopy(full), copy.deepcopy(base)
        for cfg in (a, b):
            cfg['model_params'].pop('num_gaussians')
        a['model_params'].pop('kappa')
        for key in ('experiment_name', 'method', 'k_max', 'kappa'):
            a['experiment_params'].pop(key)
        self.assertEqual(a, b)
        self.assertEqual(full['train_params']['train_epochs'], 2500)
        self.assertEqual(full['experiment_params']['seed'], 2024)

    def test_smoke_config_small(self):
        smoke = self.load(CONF / 'smoke_kappa_k16_peds_imptc.json')
        self.assertLessEqual(smoke['train_params']['train_epochs'], 3)
        self.assertEqual(smoke['model_params']['num_gaussians'], 16)

    def test_full_run_requires_passed_smoke(self):
        from kappa_mdn import train
        from unittest import mock
        with mock.patch.object(train, 'SMOKE_REPORT', ROOT / 'kappa_mdn/reports/does_not_exist.json'):
            with self.assertRaises(RuntimeError):
                train.check_full_allowed()


class HashGuardTest(unittest.TestCase):
    def test_snapshot_matches_and_detects_change(self):
        base_hashes.verify()
        snapshot = base_hashes.compute()
        key = next(iter(snapshot))
        with self.assertRaises(RuntimeError):
            base_hashes.verify(dict(snapshot, **{key: '0' * 64}))


if __name__ == '__main__':
    unittest.main()
