import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from growing_mdn import base_hashes

CONFIG_DIR = ROOT / 'growing_mdn/configs/imptc'


class ConfigFilesTest(unittest.TestCase):
    def load(self, name):
        return json.loads((CONFIG_DIR / name).read_text())

    def test_full_config_matches_spec(self):
        cfg = self.load('growing_peds_imptc.json')
        g = cfg['experiment_params']['growth']
        self.assertEqual((g['initial_k'], g['k_max'], g['phase0_epochs'], g['phase_epochs']), (1, 12, 400, 200))
        self.assertEqual(g['split_delta_relative'], 0.05)
        self.assertEqual(g['restart_lr_factor'], 0.1)
        self.assertEqual(g['max_split_nll_change'], 0.01)
        self.assertNotIn('split_delta', g)
        self.assertEqual(cfg['train_params']['train_epochs'], 400 + 11 * 200)
        self.assertEqual(cfg['model_params']['num_gaussians'], 1)
        self.assertEqual(cfg['model_params']['mdn_parameterization'], {'mode': 'legacy'})
        self.assertEqual(cfg['train_params']['eval_data_reduction'], 1.0)
        self.assertEqual(cfg['train_params']['train_data_reduction'], 0.5)
        self.assertEqual(cfg['train_params']['batch_size'], 4096)
        self.assertEqual(cfg['train_params']['lr_default'], 1e-3)
        self.assertEqual(cfg['experiment_params']['seed'], 2024)
        self.assertEqual(set(cfg['experiment_params']['selection']),
                         {'epsilon_nll', 'ravg_tolerance_pp', 'sharpness_tolerance_ratio'})

    def test_smoke_config_is_small(self):
        cfg = self.load('smoke_growing_peds_imptc.json')
        g = cfg['experiment_params']['growth']
        total = g['phase0_epochs'] + (g['k_max'] - g['initial_k']) * g['phase_epochs']
        self.assertEqual(cfg['train_params']['train_epochs'], total)
        self.assertLessEqual(total, 6)
        self.assertEqual(g['split_delta_relative'], 0.05)
        self.assertEqual(g['restart_lr_factor'], 0.1)
        self.assertEqual(g['max_split_nll_change'], 0.05)
        self.assertNotIn('split_delta', g)

    def test_all_other_baseline_fields_unchanged(self):
        base = json.loads((ROOT / 'base_mdn/configs/imptc/default_peds_imptc.json').read_text())
        cfg = self.load('growing_peds_imptc.json')
        for key in ('paths', 'eval_metrics', 'test_params'):
            self.assertEqual(cfg[key], base[key])
        for key in ('delta_t', 'max_input_horizon', 'forecast_horizon', 'lstm_input_shape',
                    'lstm_hidden_size', 'lstm_num_layers', 'output_factor'):
            self.assertEqual(cfg['model_params'][key], base['model_params'][key])
        for key in ('lr_default', 'lr_start_factor', 'lr_end_factor', 'batch_size',
                    'randomize_train_data', 'dynamic_input_horizon'):
            self.assertEqual(cfg['train_params'][key], base['train_params'][key])

    def base(self):
        return json.loads((ROOT / 'base_mdn/configs/imptc/default_peds_imptc.json').read_text())

    def test_full_config_extra_baseline_fields(self):
        base, cfg = self.base(), self.load('growing_peds_imptc.json')
        tp, ep = cfg['train_params'], cfg['experiment_params']
        self.assertEqual(tp['train_epochs'], tp['eval_epoch_step'])
        self.assertEqual(tp['min_dynamic_input_horizon'], base['train_params']['min_dynamic_input_horizon'])
        self.assertIs(tp['resume_training'], False)
        self.assertEqual(ep['fixed_sample_manifest'], base['experiment_params']['fixed_sample_manifest'])
        self.assertEqual(ep['checkpoint_epochs'], [1, 5, 10])
        self.assertEqual(ep['checkpoint_every'], 100)
        self.assertEqual(ep['save_last_every'], 1)
        self.assertEqual(ep['metric_every'], 0)

    def test_smoke_config_baseline_fields(self):
        base, cfg = self.base(), self.load('smoke_growing_peds_imptc.json')
        g, tp, ep = cfg['experiment_params']['growth'], cfg['train_params'], cfg['experiment_params']
        self.assertEqual(cfg['model_params']['num_gaussians'], 1)
        self.assertEqual(cfg['model_params']['mdn_parameterization'], {'mode': 'legacy'})
        self.assertEqual((g['initial_k'], g['k_max']), (1, 3))
        self.assertEqual(tp['train_epochs'], 4)
        self.assertEqual(tp['eval_epoch_step'], 4)
        self.assertEqual(tp['eval_data_reduction'], 1.0)
        self.assertEqual(ep['seed'], 2024)
        self.assertEqual(ep['fixed_sample_manifest'], base['experiment_params']['fixed_sample_manifest'])
        for key in ('lr_default', 'lr_start_factor', 'lr_end_factor', 'randomize_train_data',
                    'dynamic_input_horizon', 'min_dynamic_input_horizon', 'resume_training'):
            self.assertEqual(tp[key], base['train_params'][key])
        for key in ('paths', 'eval_metrics'):
            self.assertEqual(cfg[key], base[key])
        for key in ('delta_t', 'max_input_horizon', 'forecast_horizon', 'lstm_input_shape',
                    'lstm_hidden_size', 'lstm_num_layers', 'output_factor'):
            self.assertEqual(cfg['model_params'][key], base['model_params'][key])
        self.assertEqual(set(ep['selection']),
                         {'epsilon_nll', 'ravg_tolerance_pp', 'sharpness_tolerance_ratio'})


class HashGuardTest(unittest.TestCase):
    def test_current_base_matches_snapshot(self):
        base_hashes.verify()

    def test_detects_change(self):
        snapshot = base_hashes.compute()
        key = next(iter(snapshot))
        snapshot[key] = '0' * 64
        with self.assertRaises(RuntimeError):
            base_hashes.verify(expected=snapshot)

    def test_detects_added_file(self):
        snapshot = base_hashes.compute()
        snapshot['base_mdn/phantom_file.py'] = '1' * 64
        with self.assertRaises(RuntimeError):
            base_hashes.verify(expected=snapshot)

    def test_detects_removed_file(self):
        snapshot = base_hashes.compute()
        del snapshot[next(iter(snapshot))]
        with self.assertRaises(RuntimeError):
            base_hashes.verify(expected=snapshot)


class CheckConfigTest(unittest.TestCase):
    def cfg(self, **overrides):
        import copy
        from types import SimpleNamespace
        raw = json.loads((CONFIG_DIR / 'growing_peds_imptc.json').read_text())
        for section, values in overrides.items():
            raw[section].update(values)
        return SimpleNamespace(model_params=raw['model_params'], train_params=raw['train_params'],
                               experiment_params=raw['experiment_params'])

    def test_valid_config(self):
        from growing_mdn.train import check_config
        self.assertEqual(check_config(self.cfg())['k_max'], 12)

    def test_rejects_misuse(self):
        from growing_mdn.train import check_config
        base = json.loads((CONFIG_DIR / 'growing_peds_imptc.json').read_text())['experiment_params']['growth']
        bad = [
            {'train_params': {'train_epochs': 2500}},
            {'train_params': {'eval_data_reduction': 0.5}},
            {'train_params': {'dynamic_input_horizon': True}},
            {'model_params': {'num_gaussians': 3}},
            {'model_params': {'mdn_parameterization': {'mode': 'paper'}}},
            {'experiment_params': {'growth': dict(base, k_max=1)}},
            {'experiment_params': {'growth': dict(base, split_delta_relative=0.0)}},
            {'experiment_params': {'growth': dict(base, split_delta_relative=0.6)}},
            {'experiment_params': {'growth': dict(base, restart_lr_factor=0.0)}},
            {'experiment_params': {'growth': dict(base, restart_lr_factor=-0.1)}},
            {'experiment_params': {'growth': dict(base, restart_lr_factor=1.5)}},
            {'experiment_params': {'growth': dict(base, max_split_nll_change=0.0)}},
            {'experiment_params': {'growth': dict(base, max_split_nll_change=-1)}},
        ]
        for key in ('split_delta_relative', 'restart_lr_factor', 'max_split_nll_change'):
            missing = {k: v for k, v in base.items() if k != key}
            bad.append({'experiment_params': {'growth': missing}})
        old_only = {k: v for k, v in base.items() if k != 'split_delta_relative'}
        old_only['split_delta'] = 0.05
        bad.append({'experiment_params': {'growth': old_only}})
        for overrides in bad:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    check_config(self.cfg(**overrides))


if __name__ == '__main__':
    unittest.main()
