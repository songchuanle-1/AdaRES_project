"""Reject an unexecuted/unchanged/invalid native containment response."""
import ast
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

BODY = Path(__file__).with_name('probe_y64_native_smoke.py').read_text(encoding='utf-8')
TREE = ast.parse(BODY, feature_version=(3, 8))
functions = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in
             ('check_path_calls', 'check_response_rows')]
NAMESPACE = {}
exec(compile(ast.Module(body=functions, type_ignores=[]), '<native-guards>', 'exec'), NAMESPACE)


class NativeContainmentGuards(unittest.TestCase):
    def rows(self):
        return [dict(containment_weight_nonzero=i >= 7, context_abs_max=1.,
                     logit_delta_max=.01 if i >= 7 else 0.) for i in range(14)]

    def test_complete_two_phase_response(self):
        NAMESPACE['check_path_calls']([True] * 7, 7)
        NAMESPACE['check_response_rows'](self.rows())

    def test_missing_extra_and_detached_stage_rejected(self):
        for norms, calls in [([True] * 6, 6), ([True] * 8, 8), ([False] * 7, 7), ([True] * 7, 6)]:
            with self.subTest(norms=norms, calls=calls), self.assertRaises(AssertionError):
                NAMESPACE['check_path_calls'](norms, calls)

    def test_missing_second_step_rejected(self):
        with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](self.rows()[:7])

    def test_nonzero_initial_anchor_rejected(self):
        for field, value in [('containment_weight_nonzero', True), ('logit_delta_max', .01)]:
            rows = self.rows()
            rows[0][field] = value
            with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](rows)

    def test_zero_weight_or_unexecuted_response_rejected(self):
        for field, value in [('containment_weight_nonzero', False), ('logit_delta_max', 0.), ('context_abs_max', 0.)]:
            rows = self.rows()
            for row in rows[7:]: row[field] = value
            with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](rows)

    def test_nan_inf_negative_or_nonboolean_evidence_rejected(self):
        for field, value in [('logit_delta_max', float('nan')), ('context_abs_max', float('inf')),
                             ('logit_delta_max', -.1), ('containment_weight_nonzero', 1)]:
            rows = self.rows()
            rows[7][field] = value
            with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](rows)

    def test_wrong_gpu_refused_before_remote_queries(self):
        for gpu in (None, 1):
            with self.subTest(gpu=gpu), self.assertRaises(AssertionError):
                exec(compile(BODY, '<y64-native>', 'exec'), dict(P=dict(physical_gpu=gpu)))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}), self.assertRaises(AssertionError):
            exec(compile(BODY, '<y64-native>', 'exec'), dict(P=dict(physical_gpu=0)))


if __name__ == '__main__': unittest.main()
