import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sparsemax_mdn import base_hashes  # noqa: E402

CONF = ROOT / 'sparsemax_mdn/configs/imptc'
BASE = ROOT / 'base_mdn/configs/imptc'


def load(path):
    return json.loads(path.read_text())


class ConfigTest(unittest.TestCase):
    def test_full_differs_from_baseline_only_in_k_and_experiment_params(self):
        full = load(CONF / 'sparsemax_k8_peds_imptc.json')
        base = load(BASE / 'default_peds_imptc.json')
        self.assertEqual(full['model_params']['num_gaussians'], 8)
        self.assertEqual(full['experiment_params']['k_max'], 8)
        self.assertEqual(full['experiment_params']['method'], 'sparsemax_pi')
        self.assertEqual(full['experiment_params']['experiment_name'], 'imptc_sparsemax_mdn')
        a, b = copy.deepcopy(full), copy.deepcopy(base)
        for cfg in (a, b):
            cfg['model_params'].pop('num_gaussians')
        for key in ('experiment_name', 'method', 'k_max'):
            a['experiment_params'].pop(key)
        self.assertEqual(a, b)
        self.assertEqual(full['train_params']['train_epochs'], 2500)
        self.assertEqual(full['experiment_params']['seed'], 2024)

    def test_k16_full_config_differs_from_k8_only_in_cap(self):
        k8 = load(CONF / 'sparsemax_k8_peds_imptc.json')
        k16 = load(CONF / 'sparsemax_k16_peds_imptc.json')
        self.assertEqual(k16['model_params']['num_gaussians'], 16)
        self.assertEqual(k16['experiment_params']['k_max'], 16)
        for cfg in (k8, k16):
            cfg['model_params'].pop('num_gaussians')
            cfg['experiment_params'].pop('k_max')
        self.assertEqual(k8, k16)

    def test_smoke_config_small(self):
        smoke = load(CONF / 'smoke_sparsemax_k8_peds_imptc.json')
        self.assertLessEqual(smoke['train_params']['train_epochs'], 3)
        self.assertEqual(smoke['model_params']['num_gaussians'], 8)
        self.assertEqual(smoke['experiment_params']['metric_every'], 1)
        self.assertEqual(smoke['model_params'].get('mdn_parameterization', {'mode': 'legacy'}), {'mode': 'legacy'})


class HashGuardTest(unittest.TestCase):
    def test_snapshot_matches_current_base_mdn(self):
        base_hashes.verify()

    def test_guard_detects_change(self):
        snapshot = base_hashes.compute()
        key = next(iter(snapshot))
        tampered = dict(snapshot, **{key: '0' * 64})
        with self.assertRaises(RuntimeError):
            base_hashes.verify(tampered)
        with self.assertRaises(RuntimeError):
            base_hashes.verify({k: v for k, v in snapshot.items() if k != key})


class FullGuardTest(unittest.TestCase):
    def test_full_requires_passed_smoke(self):
        from sparsemax_mdn import train
        from unittest import mock
        with mock.patch.object(train, 'SMOKE_REPORT', ROOT / 'sparsemax_mdn/reports/does_not_exist.json'):
            with self.assertRaises(RuntimeError):
                train.check_full_allowed()


if __name__ == '__main__':
    unittest.main()
