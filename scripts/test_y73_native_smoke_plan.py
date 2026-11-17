"""Reject missing, detached, mismatched or fabricated initial-query evidence."""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name('probe_y73_native_smoke.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE, feature_version=(3, 8)); N = {}
FUNCTIONS = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and
             n.name in ('check_path_calls', 'check_native_inputs', 'check_response_rows')]
assert len(FUNCTIONS) == 3
exec(compile(ast.Module(body=FUNCTIONS, type_ignores=[]), '<initial-query-guards>', 'exec'), N)


class Guards(unittest.TestCase):
    def rows(self):
        return [dict(stage=i % 6, weight_nonzero=i >= 6, expressions=8, queries=128,
            current_qqa_ffn_identity_verified=True, initial_query_identity_verified=True,
            undetached_anchor_verified=True, prediction_consumption_verified=True,
            static_mask_features_preserved=True, padding_identity_verified=True,
            reference_expression_rows=[0, 7], reference_queries=[0, 127], reference_channels=[0, 127, 255],
            reference_error_max=1e-7, query_delta_max=.001 if i >= 6 else 0.,
            anchor_current_difference_max=4.) for i in range(12)]

    def test_valid_schema_only(self):
        N['check_path_calls'](list(range(6)), list(range(6)), list(range(7)), 7, [True] * 7)
        N['check_response_rows'](self.rows())

    def test_wrong_current_or_initial_anchor_identity(self):
        query, anchor = object(), object(); original = [query, query, anchor, anchor]
        N['check_native_inputs'](*original)
        for i in range(4):
            args = list(original); args[i] = object()
            with self.assertRaises(AssertionError): N['check_native_inputs'](*args)

    def test_missing_extra_reordered_or_noninteger_paths(self):
        for position, count in ((0, 6), (1, 6), (2, 7)):
            for bad in (list(range(count - 1)), list(range(count + 1)),
                        list(reversed(range(count))), [float(i) for i in range(count)]):
                args = [list(range(6)), list(range(6)), list(range(7)), 7, [True] * 7]; args[position] = bad
                with self.subTest(args=args), self.assertRaises(AssertionError): N['check_path_calls'](*args)

    def test_missing_detached_or_nonboolean_head_paths(self):
        for count, norms in ((6, [True] * 7), (8, [True] * 7), (7., [True] * 7),
                             (7, [False] * 7), (7, [1] * 7), (7, [True] * 6)):
            with self.assertRaises(AssertionError):
                N['check_path_calls'](list(range(6)), list(range(6)), list(range(7)), count, norms)

    def test_missing_second_step_or_extra_measurements(self):
        for rows in (self.rows()[:6], self.rows() + self.rows()[:1]):
            with self.assertRaises(AssertionError): N['check_response_rows'](rows)

    def test_nonzero_initial_anchor_residual(self):
        for key, value in (('weight_nonzero', True), ('query_delta_max', .1)):
            rows = self.rows(); rows[0][key] = value
            with self.assertRaises(AssertionError): N['check_response_rows'](rows)

    def test_missing_second_weight_response_or_anchor_difference(self):
        for key, value in (('weight_nonzero', False), ('query_delta_max', 0.), ('anchor_current_difference_max', 0.)):
            rows = self.rows(); rows[6][key] = value
            with self.assertRaises(AssertionError): N['check_response_rows'](rows)

    def test_nonfinite_or_negative_measurements(self):
        for key in ('reference_error_max', 'query_delta_max', 'anchor_current_difference_max'):
            for value in (float('nan'), float('inf'), -.1):
                rows = self.rows(); rows[8][key] = value
                with self.assertRaises(AssertionError): N['check_response_rows'](rows)

    def test_detached_bypassed_wrong_features_or_nonboolean_flags(self):
        for key in ('current_qqa_ffn_identity_verified', 'initial_query_identity_verified', 'undetached_anchor_verified',
                    'prediction_consumption_verified', 'static_mask_features_preserved', 'padding_identity_verified'):
            for value in (False, 1):
                rows = self.rows(); rows[7][key] = value
                with self.assertRaises(AssertionError): N['check_response_rows'](rows)
        rows = self.rows(); rows[7]['weight_nonzero'] = 1
        with self.assertRaises(AssertionError): N['check_response_rows'](rows)

    def test_wrong_reference_rows_queries_channels_or_stage(self):
        for key, value in (('reference_expression_rows', [0, 6]), ('expressions', 8.), ('queries', 128.),
                           ('reference_queries', [0, 126]), ('reference_channels', [0, 127, 254]), ('stage', 0)):
            rows = self.rows(); rows[7][key] = value
            with self.assertRaises(AssertionError): N['check_response_rows'](rows)

    def test_native_train_signature_unchanged(self):
        calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'native' and n.func.attr == 'train']
        self.assertEqual(len(calls), 1)
        self.assertEqual([ast.unparse(n) for n in calls[0].args],
                         ['1', 'model', 'FirstTwoBatches()', 'optimizer', 'scheduler', 'cfg', 'logger', 'Writer()'])


if __name__ == '__main__': unittest.main()
