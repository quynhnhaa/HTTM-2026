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
from shared_decoder_mdn import base_hashes  # noqa: E402
from shared_decoder_mdn.model import SharedDecoderMDN, time_features  # noqa: E402

K, T = 3, 48
CFG = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'num_gaussians': K, 'output_factor': 6,
       'forecast_horizon': T, 'lstm_num_layers': 1, 'decoder': {'hidden': 64, 'num_frequencies': 4}}
CONF = ROOT / 'shared_decoder_mdn/configs/imptc'
BASE = ROOT / 'base_mdn/configs/imptc'


class ModelTest(unittest.TestCase):
    def test_output_has_baseline_layout_and_decodes(self):
        torch.manual_seed(0)
        out = SharedDecoderMDN(CFG)(torch.randn(5, 32, 4))
        self.assertEqual(out.shape, (5, T, 6 * K))
        d = decode_mdn_output(out, K)
        torch.testing.assert_close(d['pi'].sum(-1), torch.ones(5, T))
        self.assertTrue(bool((d['sigma'] > 0).all()))

    def test_lstm_matches_baseline_architecture(self):
        base = LSTM_Trajectory_Forecast(CFG)
        new = SharedDecoderMDN(CFG)
        self.assertEqual({n: p.shape for n, p in base.lstm.named_parameters()},
                         {n: p.shape for n, p in new.lstm.named_parameters()})
        self.assertFalse(hasattr(new, 'fc'))

    def test_fewer_parameters_than_baseline_decoder(self):
        base = sum(p.numel() for p in LSTM_Trajectory_Forecast(CFG).fc.parameters())
        new = sum(p.numel() for p in SharedDecoderMDN(CFG).decoder.parameters())
        self.assertLess(new, base)

    def test_weights_are_shared_so_steps_differ_only_through_time_features(self):
        torch.manual_seed(1)
        m = SharedDecoderMDN(CFG).eval()
        x = torch.randn(2, 32, 4)
        with torch.no_grad():
            out = m(x)
            state = m.lstm(x)[0][:, -1, :]
            for t in (0, 17, 47):
                joint = torch.cat([state, m.step_features[t][None].expand(2, -1)], -1)
                torch.testing.assert_close(out[:, t], m.decoder(joint))

    def test_time_features_are_smooth_and_bounded(self):
        f = time_features(T, 4)
        self.assertEqual(f.shape, (T, 9))
        self.assertLessEqual(float((f[1:] - f[:-1]).abs().max()), 0.3)
        self.assertLessEqual(float(f.abs().max()), 1.0)

    def test_baseline_loss_is_finite_and_all_parameters_get_gradient(self):
        torch.manual_seed(2)
        m = SharedDecoderMDN(CFG)
        loss, diverged = NLL_MDN_loss(m(torch.randn(8, 32, 4)), torch.randn(8, T, 2), K, None)
        self.assertFalse(diverged)
        loss.backward()
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)


class ConfigTest(unittest.TestCase):
    def test_full_config_differs_from_baseline_only_in_declared_keys(self):
        full = json.loads((CONF / 'shared_decoder_peds_imptc.json').read_text())
        base = json.loads((BASE / 'default_peds_imptc.json').read_text())
        self.assertEqual(full['model_params'].pop('decoder'), {'hidden': 64, 'num_frequencies': 4})
        for key in ('experiment_name', 'method'):
            full['experiment_params'].pop(key)
            base['experiment_params'].pop(key, None)
        self.assertEqual(full, base)


class H64ConfigTest(unittest.TestCase):
    def test_h64_config_differs_from_first_run_only_in_lstm_hidden(self):
        h8 = json.loads((CONF / 'shared_decoder_peds_imptc.json').read_text())
        h64 = json.loads((CONF / 'shared_decoder_h64_peds_imptc.json').read_text())
        self.assertEqual(h64['model_params'].pop('lstm_hidden_size'), 64)
        self.assertEqual(h8['model_params'].pop('lstm_hidden_size'), 8)
        for cfg in (h8, h64):
            cfg['experiment_params'].pop('experiment_name')
        self.assertEqual(h8, h64)

    def test_h32_config_differs_from_h64_only_in_lstm_hidden(self):
        h64 = json.loads((CONF / 'shared_decoder_h64_peds_imptc.json').read_text())
        h32 = json.loads((CONF / 'shared_decoder_h32_peds_imptc.json').read_text())
        self.assertEqual(h32['model_params'].pop('lstm_hidden_size'), 32)
        self.assertEqual(h64['model_params'].pop('lstm_hidden_size'), 64)
        for cfg in (h32, h64):
            cfg['experiment_params'].pop('experiment_name')
        self.assertEqual(h32, h64)

    def test_model_builds_and_runs_with_hidden_64(self):
        cfg = dict(CFG, lstm_hidden_size=64)
        out = SharedDecoderMDN(cfg)(torch.randn(3, 32, 4))
        self.assertEqual(out.shape, (3, T, 6 * K))


class EvaluateGuardTest(unittest.TestCase):
    def test_guards(self):
        from shared_decoder_mdn.evaluate import check_request, LOCK
        import tempfile
        check_request('validation', False)
        with self.assertRaises(ValueError):
            check_request('validation', True)
        with self.assertRaises(ValueError):
            check_request('test', False)
        with self.assertRaises(ValueError):
            check_request('bogus', False)
        with tempfile.TemporaryDirectory() as d:
            check_request('test', True, d)
            (Path(d) / LOCK).write_text('{}')
            with self.assertRaises(ValueError):
                check_request('test', True, d)


class HashGuardTest(unittest.TestCase):
    def test_base_unchanged(self):
        base_hashes.verify()


if __name__ == '__main__':
    unittest.main()
