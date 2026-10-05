import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from growing_mdn.select_k import build_records, check_run_ready, check_selection_allowed, frozen_rule
from growing_mdn.selection import select_k
from growing_mdn.evaluate import check_official_request

METRICS = {'ravg_percent': 97.0, 'rmin_percent': 94.0, 's68_m2_per_s': 1.0, 's95_m2_per_s': 5.0}


class BuildRecordsTest(unittest.TestCase):
    def test_records_sorted_and_complete(self):
        summary = {'phases': {
            '2': {'k': 2, 'validation_nll': -0.9, 'metrics': dict(METRICS)},
            '1': {'k': 1, 'validation_nll': -0.3, 'metrics': dict(METRICS)}}}
        records = build_records(summary)
        self.assertEqual([r['k'] for r in records], [1, 2])
        self.assertEqual(select_k(records, 0.0, 0.0, 0.0)['selected_k'], 2)

    def test_missing_metric_rejected(self):
        summary = {'phases': {'1': {'k': 1, 'validation_nll': -0.3, 'metrics': {'ravg_percent': 97.0}}}}
        with self.assertRaises(KeyError):
            build_records(summary)


class OfficialGuardTest(unittest.TestCase):
    def test_guard(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'evaluation_k03.json'
            with self.assertRaises(ValueError):
                check_official_request(3, None, out)
            with self.assertRaises(ValueError):
                check_official_request(2, {'selected_k': 3}, out)
            check_official_request(3, {'selected_k': 3}, out)
            out.write_text('{}')
            with self.assertRaises(ValueError):
                check_official_request(3, {'selected_k': 3}, out)


class SelectGuardTest(unittest.TestCase):
    CFG = {'experiment_params': {'growth': {'initial_k': 1, 'k_max': 3},
                                 'selection': {'epsilon_nll': 0.02, 'ravg_tolerance_pp': 0.5,
                                               'sharpness_tolerance_ratio': 0.1}}}

    def test_ready(self):
        phases = {str(k): {} for k in (1, 2, 3)}
        check_run_ready({'status': 'completed'}, self.CFG, {'phases': phases})
        with self.assertRaises(ValueError) as ctx:
            check_run_ready({'status': 'completed'}, self.CFG, {'phases': {'1': {}, '3': {}}})
        self.assertIn('[2]', str(ctx.exception))
        with self.assertRaises(ValueError):
            check_run_ready({'status': 'running'}, self.CFG, {'phases': phases})

    def test_existing_selection_and_official_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            check_selection_allowed(run)
            (run / 'testing').mkdir()
            (run / 'testing/evaluation_k03_validation_limited.json').write_text('{}')
            check_selection_allowed(run)
            (run / 'testing/evaluation_k03.json').write_text('{}')
            with self.assertRaises(ValueError):
                check_selection_allowed(run)
            (run / 'testing/evaluation_k03.json').unlink()
            (run / 'selection.json').write_text('{}')
            with self.assertRaises(ValueError):
                check_selection_allowed(run)

    def test_tolerances_from_resolved_config(self):
        self.assertEqual(frozen_rule(self.CFG)['epsilon_nll'], 0.02)


class OfficialGuardExtraTest(unittest.TestCase):
    def test_other_k_result_and_run_id(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'evaluation_k03.json'
            sel = {'selected_k': 3, 'run_id': 'r'}
            self.assertEqual(check_official_request(None, sel, out, 'r'), 3)
            with self.assertRaises(ValueError):
                check_official_request(3, sel, out, 'other')
            (Path(tmp) / 'evaluation_k02.json').write_text('{}')
            with self.assertRaises(ValueError):
                check_official_request(3, sel, out, 'r')
            with self.assertRaises(ValueError):
                check_official_request(None, None, out)


if __name__ == '__main__':
    unittest.main()
