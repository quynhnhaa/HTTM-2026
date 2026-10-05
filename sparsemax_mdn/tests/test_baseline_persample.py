import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / 'base_mdn')):
    if p not in sys.path:
        sys.path.insert(0, p)
from sparsemax_mdn import baseline_persample as bp  # noqa: E402


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'K3'

    def tearDown(self):
        self.tmp.cleanup()

    def test_validation_ok(self):
        bp.check_baseline_request('validation', False, None, self.out)
        bp.check_baseline_request('validation', False, 300, self.out)

    def test_test_needs_confirmation(self):
        with self.assertRaisesRegex(ValueError, 'confirm-test-baseline'):
            bp.check_baseline_request('test', False, None, self.out)
        bp.check_baseline_request('test', True, None, self.out)

    def test_test_rejects_limit(self):
        with self.assertRaisesRegex(ValueError, 'limit'):
            bp.check_baseline_request('test', True, 10, self.out)

    def test_confirm_on_validation_rejected_and_bad_split(self):
        with self.assertRaises(ValueError):
            bp.check_baseline_request('validation', True, None)
        for bad in ('TEST', 'testing', '', None):
            with self.assertRaises(ValueError):
                bp.check_baseline_request(bad, True, None)
        with self.assertRaises(ValueError):
            bp.check_baseline_request('validation', False, 0)

    def test_lock_and_npz_block_second_test(self):
        bp.check_baseline_request('test', True, None, self.out)
        bp.claim_test(self.out, {})
        with self.assertRaisesRegex(ValueError, 'already exists'):
            bp.check_baseline_request('test', True, None, self.out)
        with self.assertRaises(FileExistsError):
            bp.claim_test(self.out, {})
        (self.out / bp.TEST_CLAIM).unlink()
        bp.check_baseline_request('test', True, None, self.out)
        (self.out / 'test_best_persample.npz').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            bp.check_baseline_request('test', True, None, self.out)

    def test_validation_files_do_not_block_test(self):
        self.out.mkdir()
        (self.out / 'validation_best_persample.npz').write_bytes(b'')
        bp.check_baseline_request('test', True, None, self.out)

    def test_refusal_happens_before_any_data_or_checkpoint_read(self):
        boom = AssertionError('data/checkpoint touched before guard')
        with mock.patch.object(bp, 'DataLoader', side_effect=boom), \
                mock.patch.object(bp.torch, 'load', side_effect=boom), \
                mock.patch.object(bp, 'ConfigLoader', side_effect=boom), \
                mock.patch.object(bp.base_hashes, 'verify', side_effect=boom):
            with self.assertRaisesRegex(ValueError, 'confirm-test-baseline'):
                bp.main(['--baseline-run', '/nonexistent', '--label', 'X', '--split', 'test'])
            with self.assertRaisesRegex(ValueError, 'limit'):
                bp.main(['--baseline-run', '/nonexistent', '--label', 'X', '--split', 'test',
                         '--confirm-test-baseline', '--limit', '5'])

    def test_second_test_refused_before_data_via_main(self):
        boom = AssertionError('touched')
        with mock.patch.object(bp, 'OUT_ROOT', Path(self.tmp.name)):
            bp.claim_test(Path(self.tmp.name) / 'L', {})
            with mock.patch.object(bp, 'DataLoader', side_effect=boom), mock.patch.object(bp.torch, 'load', side_effect=boom):
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    bp.main(['--baseline-run', '/x', '--label', 'L', '--split', 'test', '--confirm-test-baseline'])

    def test_argparse_choices(self):
        with self.assertRaises(SystemExit):
            with mock.patch('sys.stderr'):
                bp.build_parser().parse_args(['--baseline-run', 'x', '--label', 'a', '--split', 'TEST'])


class HelperTest(unittest.TestCase):
    def test_output_stem(self):
        self.assertEqual(bp.output_stem('validation', True), 'validation_best_limited')
        self.assertEqual(bp.output_stem('test', False), 'test_best')

    def test_select_entry(self):
        abl = {'results': [{'num_gaussians': 1}, {'num_gaussians': 3, 'x': 1}]}
        self.assertEqual(bp.select_ablation_entry(abl, 3)['x'], 1)
        with self.assertRaises(ValueError):
            bp.select_ablation_entry(abl, 5)

    def test_reproduction_diff(self):
        d = bp.reproduction_diff({'a': 11.0, 'b': 1.0, 'c': None}, {'a': 10.0, 'b': 0.0, 'c': 1.0}, ('a', 'b', 'c', 'd'))
        self.assertAlmostEqual(d['a']['absolute_diff'], 1.0)
        self.assertAlmostEqual(d['a']['relative_diff'], 0.1)
        self.assertIsNone(d['b']['relative_diff'])
        self.assertIsNone(d['c']['absolute_diff'])
        self.assertIsNone(d['d']['current'])
        neg = bp.reproduction_diff({'a': -0.9}, {'a': -1.0}, ('a',))
        self.assertAlmostEqual(neg['a']['relative_diff'], 0.1)


if __name__ == '__main__':
    unittest.main()
