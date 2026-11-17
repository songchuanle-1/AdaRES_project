"""Reject missing, bypassed, unexecuted or invalid native feedback evidence."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

BODY = Path(__file__).with_name('probe_y65_native_smoke.py').read_text(encoding='utf-8')
TREE = ast.parse(BODY, feature_version=(3, 8))
functions = [n for n in TREE.body if isinstance(n, ast.FunctionDef)
             and n.name in ('check_path_calls', 'check_response_rows')]
NAMESPACE = {}
exec(compile(ast.Module(body=functions, type_ignores=[]), '<native-feedback-guards>', 'exec'), NAMESPACE)


class NativeFeedbackGuards(unittest.TestCase):
    def rows(self):
        return [dict(direction_nonzero=i >= 6, centered_max=.2,
            query_delta_max=.01 if i >= 6 else 0., consumed_by_refinement=True) for i in range(12)]

    def test_complete_two_phase_response(self):
        NAMESPACE['check_path_calls']([True] * 7, 7, 6, 6)
        NAMESPACE['check_response_rows'](self.rows())

    def test_missing_extra_or_detached_stage_rejected(self):
        for norms, indicator, feedback, consumed in [
                ([True] * 6, 6, 6, 6), ([True] * 8, 8, 6, 6),
                ([False] * 7, 7, 6, 6), ([True] * 7, 6, 6, 6),
                ([True] * 7, 7, 5, 6), ([True] * 7, 7, 7, 6),
                ([True] * 7, 7, 6, 5)]:
            with self.subTest(values=(norms, indicator, feedback, consumed)), self.assertRaises(AssertionError):
                NAMESPACE['check_path_calls'](norms, indicator, feedback, consumed)

    def test_missing_second_step_rejected(self):
        with self.assertRaises(AssertionError):
            NAMESPACE['check_response_rows'](self.rows()[:6])

    def test_nonzero_initial_anchor_rejected(self):
        for field, value in [('direction_nonzero', True), ('query_delta_max', .01)]:
            rows = self.rows()
            rows[0][field] = value
            with self.assertRaises(AssertionError):
                NAMESPACE['check_response_rows'](rows)

    def test_zero_direction_or_unexecuted_response_rejected(self):
        for field, value in [('direction_nonzero', False), ('query_delta_max', 0.), ('centered_max', 0.)]:
            rows = self.rows()
            for row in rows[6:]:
                row[field] = value
            with self.assertRaises(AssertionError):
                NAMESPACE['check_response_rows'](rows)

    def test_nonfinite_negative_or_nonboolean_evidence_rejected(self):
        for field, value in [('query_delta_max', float('nan')), ('centered_max', float('inf')),
                             ('query_delta_max', -.1), ('direction_nonzero', 1)]:
            rows = self.rows()
            rows[6][field] = value
            with self.assertRaises(AssertionError):
                NAMESPACE['check_response_rows'](rows)

    def test_bypassed_refinement_or_nonboolean_consumption_rejected(self):
        for value in (False, 1):
            rows = self.rows()
            rows[6]['consumed_by_refinement'] = value
            with self.assertRaises(AssertionError):
                NAMESPACE['check_response_rows'](rows)

    def test_wrong_gpu_refused_before_remote_queries(self):
        for gpu in (None, 1):
            with self.subTest(gpu=gpu), self.assertRaises(AssertionError):
                exec(compile(BODY, '<y65-native>', 'exec'), dict(P=dict(physical_gpu=gpu)))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}), self.assertRaises(AssertionError):
            exec(compile(BODY, '<y65-native>', 'exec'), dict(P=dict(physical_gpu=0)))


if __name__ == '__main__':
    unittest.main()
