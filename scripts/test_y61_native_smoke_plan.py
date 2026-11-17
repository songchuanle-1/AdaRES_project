"""CPU-only guard/count tests; these never execute the native GPU smoke."""

import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch


BODY = Path(__file__).with_name('probe_y61_native_smoke.py').read_text(encoding='utf-8')
TREE = ast.parse(BODY, feature_version=(3, 8))
FUNCTION = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'check_path_calls')
NAMESPACE = {}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), '<path-count-check>', 'exec'), NAMESPACE)
check_calls = NAMESPACE['check_path_calls']


class NativePlanGuards(unittest.TestCase):
    def test_valid_seven_stages(self):
        check_calls([True, False] * 7, 7)

    def test_live_peer_input_refused(self):
        with self.assertRaises(AssertionError):
            check_calls([True, True] * 7, 7)

    def test_detached_focal_input_refused(self):
        with self.assertRaises(AssertionError):
            check_calls([False, False] * 7, 7)

    def test_missing_stage_refused(self):
        with self.assertRaises(AssertionError):
            check_calls([True, False] * 6, 6)

    def test_extra_stage_refused(self):
        with self.assertRaises(AssertionError):
            check_calls([True, False] * 8, 8)

    def test_count_mismatch_refused(self):
        with self.assertRaises(AssertionError):
            check_calls([True, False] * 7, 6)

    def test_no_gpu_and_gpu1_refused_before_imports(self):
        for gpu in (None, 1):
            with self.subTest(gpu=gpu), self.assertRaises(AssertionError):
                exec(compile(BODY, '<y61-native>', 'exec'), dict(P=dict(physical_gpu=gpu)))

    def test_wrong_visible_device_refused_before_imports(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}):
            with self.assertRaises(AssertionError):
                exec(compile(BODY, '<y61-native>', 'exec'), dict(P=dict(physical_gpu=0)))


if __name__ == '__main__':
    unittest.main()
