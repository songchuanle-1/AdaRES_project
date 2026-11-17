"""Guard against bypassed, misaligned, unexecuted or nonfinite evidence."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

BODY = Path(__file__).with_name('probe_y66_native_smoke.py').read_text(encoding='utf-8')
TREE = ast.parse(BODY, feature_version=(3, 8))
functions = [n for n in TREE.body if isinstance(n, ast.FunctionDef)
             and n.name in ('check_path_calls', 'check_response_rows', 'check_feedback_alignment')]
NAMESPACE = {}
exec(compile(ast.Module(body=functions, type_ignores=[]), '<native-peer-feedback-guards>', 'exec'), NAMESPACE)


class NativePeerFeedbackGuards(unittest.TestCase):
    def rows(self):
        return [dict(projection_nonzero=i >= 6, context_max=.2,
            query_delta_max=.01 if i >= 6 else 0., reference_error_max=1e-8,
            reference_query_rows=[0, 64, 127], consumed_by_refinement=True) for i in range(12)]

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
        for field, value in [('projection_nonzero', True), ('query_delta_max', .01)]:
            rows = self.rows()
            rows[0][field] = value
            with self.assertRaises(AssertionError):
                NAMESPACE['check_response_rows'](rows)

    def test_zero_projection_or_unexecuted_response_rejected(self):
        for field, value in [('projection_nonzero', False), ('query_delta_max', 0.), ('context_max', 0.)]:
            rows = self.rows()
            for row in rows[6:]:
                row[field] = value
            with self.assertRaises(AssertionError):
                NAMESPACE['check_response_rows'](rows)

    def test_nonfinite_negative_or_nonboolean_evidence_rejected(self):
        for field, value in [('query_delta_max', float('nan')), ('context_max', float('inf')),
                             ('reference_error_max', float('nan')), ('query_delta_max', -.1),
                             ('projection_nonzero', 1)]:
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

    def test_incomplete_formula_rows_rejected(self):
        for values in ([], [0], [0, 63, 127]):
            rows = self.rows()
            rows[6]['reference_query_rows'] = values
            with self.assertRaises(AssertionError):
                NAMESPACE['check_response_rows'](rows)

    def test_exact_prior_native_tensors_required(self):
        args = tuple(object() for _ in range(4))
        NAMESPACE['check_feedback_alignment'](*args, args)
        for index in range(4):
            wrong = list(args)
            wrong[index] = object()
            with self.assertRaises(AssertionError):
                NAMESPACE['check_feedback_alignment'](*wrong, args)
        with self.assertRaises(AssertionError):
            NAMESPACE['check_feedback_alignment'](*args, args[:3])

    def test_wrong_gpu_refused_before_remote_queries(self):
        for gpu in (None, 1):
            with self.subTest(gpu=gpu), self.assertRaises(AssertionError):
                exec(compile(BODY, '<y66-native>', 'exec'), dict(P=dict(physical_gpu=gpu)))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}), self.assertRaises(AssertionError):
            exec(compile(BODY, '<y66-native>', 'exec'), dict(P=dict(physical_gpu=0)))


if __name__ == '__main__':
    unittest.main()
