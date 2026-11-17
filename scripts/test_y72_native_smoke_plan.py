"""Reject false native-path evidence without loading torch or reserving a GPU."""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name('probe_y72_native_smoke.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE, feature_version=(3, 8))
NAMESPACE = {}
FUNCTIONS = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and
             n.name in ('check_path_calls', 'check_native_inputs', 'check_response_rows')]
assert len(FUNCTIONS) == 3
exec(compile(ast.Module(body=FUNCTIONS, type_ignores=[]), '<visual-mask-guards>', 'exec'), NAMESPACE)


class Guards(unittest.TestCase):
    def rows(self):
        return [dict(stage=i % 6, weight_nonzero=i >= 6, expressions=8,
            native_current_visual_identity_verified=True, static_mask_identity_verified=True,
            padding_identity_verified=True, prediction_consumption_verified=True,
            padding_rows_unchanged=True, reference_expression_rows=[0, 7],
            reference_valid_feature_counts=[513, 700], reference_feature_indices=[[0, 256, 512], [0, 350, 699]],
            reference_channels=[0, 127, 255], reference_error_max=1e-7,
            feature_delta_max=.001 if i >= 6 else 0., visual_abs_max=4., padded_rows=100)
            for i in range(12)]

    def test_valid_evidence_shape_only(self):
        NAMESPACE['check_path_calls'](list(range(6)), list(range(6)), list(range(7)), 7, [True] * 7)
        NAMESPACE['check_response_rows'](self.rows())

    def test_wrong_stage_accumulated_features_and_copied_padding(self):
        features, visual, padded = object(), object(), object()
        values = [features, features, visual, visual, padded, padded]
        NAMESPACE['check_native_inputs'](*values)
        for index in range(6):
            args = list(values); args[index] = object()
            with self.subTest(index=index), self.assertRaises(AssertionError):
                NAMESPACE['check_native_inputs'](*args)

    def test_missing_extra_reordered_or_noninteger_stages(self):
        for position, count in ((0, 6), (1, 6), (2, 7)):
            for bad in (list(range(count - 1)), list(range(count + 1)),
                        list(reversed(range(count))), [float(i) for i in range(count)]):
                args = [list(range(6)), list(range(6)), list(range(7)), 7, [True] * 7]
                args[position] = bad
                with self.subTest(position=position, bad=bad), self.assertRaises(AssertionError):
                    NAMESPACE['check_path_calls'](*args)

    def test_missing_detached_or_nonboolean_shared_head_paths(self):
        for count, norms in ((6, [True] * 7), (8, [True] * 7), (7., [True] * 7),
                             (7, [False] * 7), (7, [1] * 7), (7, [True] * 6)):
            with self.assertRaises(AssertionError):
                NAMESPACE['check_path_calls'](list(range(6)), list(range(6)), list(range(7)), count, norms)

    def test_missing_second_step_or_extra_measurement(self):
        for rows in (self.rows()[:6], self.rows() + self.rows()[:1]):
            with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](rows)

    def test_nonzero_initial_anchor(self):
        for key, value in (('weight_nonzero', True), ('feature_delta_max', .1)):
            rows = self.rows(); rows[0][key] = value
            with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](rows)

    def test_missing_second_step_weight_response_or_visual(self):
        for key, value in (('weight_nonzero', False), ('feature_delta_max', 0.), ('visual_abs_max', 0.)):
            rows = self.rows(); rows[6][key] = value
            with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](rows)

    def test_nonfinite_negative_and_invalid_padding_measurements(self):
        for key in ('reference_error_max', 'feature_delta_max', 'visual_abs_max'):
            for value in (float('nan'), float('inf'), -.1):
                rows = self.rows(); rows[8][key] = value
                with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](rows)
        for value in (-1, 1.0, True):
            rows = self.rows(); rows[8]['padded_rows'] = value
            with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](rows)

    def test_wrong_identity_bypass_padding_and_nonboolean_flags(self):
        for key in ('native_current_visual_identity_verified', 'static_mask_identity_verified',
                    'padding_identity_verified', 'prediction_consumption_verified', 'padding_rows_unchanged'):
            for value in (False, 1):
                rows = self.rows(); rows[7][key] = value
                with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](rows)
        rows = self.rows(); rows[7]['weight_nonzero'] = 1
        with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](rows)

    def test_wrong_reference_expressions_features_channels_or_stage(self):
        for key, value in (('reference_expression_rows', [0, 6]), ('expressions', 8.),
                           ('reference_valid_feature_counts', [512, 700]),
                           ('reference_feature_indices', [[0, 255, 512], [0, 350, 699]]),
                           ('reference_channels', [0, 127, 254]), ('stage', 0)):
            rows = self.rows(); rows[7][key] = value
            with self.assertRaises(AssertionError): NAMESPACE['check_response_rows'](rows)

    def test_native_train_signature_unchanged(self):
        calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and
                 n.func.value.id == 'native' and n.func.attr == 'train']
        self.assertEqual(len(calls), 1)
        self.assertEqual([ast.unparse(n) for n in calls[0].args],
                         ['1', 'model', 'FirstTwoBatches()', 'optimizer', 'scheduler', 'cfg', 'logger', 'Writer()'])


if __name__ == '__main__':
    unittest.main()
