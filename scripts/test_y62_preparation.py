"""Preparation guards; synthetic metadata never substitutes for native smoke."""

import ast
import copy
import unittest

from prepare_y62_overlap_density import ROOT, validate_cpu_result


class PreparationTests(unittest.TestCase):
    def setUp(self):
        controls = [dict(dtype=dtype, focal_density_difference=.1,
            zero_anchor_output_and_input_gradients_exact=True,
            finite_parameter_gradients=True, density_weight_gradient_nonzero=True,
            permutation_padding_singleton_empty_controls=True, unique_masks_in_both_arms=True,
            focal_max_overlap_matched=True, focal_W2_output_matched_within_arithmetic=True,
            explicit_path_response_observed=True) for dtype in ('torch.float32', 'torch.float64')]
        self.result = dict(complete=True, id='AdaRES-Y62', parameters=164629639,
            additional_parameters=2, decoder_seven_calls=7, component_controls=controls)
        for key in ('implementation_test_passed', 'full_model_CPU_constructed',
                    'configured_TRAIN_data_collated', 'released_backbone_loaded',
                    'native_YAML_semantically_unchanged', 'full_decoder_CPU_executed',
                    'decoder_anchor_all_stages_exact', 'decoder_backward_finite',
                    'decoder_density_gradient_nonzero'):
            self.result[key] = True
        for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward',
                    'native_sparse_forward', 'validation_used', 'admission'):
            self.result[key] = False

    def test_fixture_schema(self):
        validate_cpu_result(self.result)

    def test_missing_dtype(self):
        self.result['component_controls'].pop()
        with self.assertRaises(AssertionError):
            validate_cpu_result(self.result)

    def test_no_new_information_witness(self):
        self.result['component_controls'][0]['focal_density_difference'] = 0
        with self.assertRaises(AssertionError):
            validate_cpu_result(self.result)

    def test_wrong_parameter_budget(self):
        self.result['parameters'] += 1
        with self.assertRaises(AssertionError):
            validate_cpu_result(self.result)

    def test_CPU_not_native_smoke(self):
        self.result['native_smoke_passed'] = True
        with self.assertRaises(AssertionError):
            validate_cpu_result(self.result)

    def test_no_decoder_path(self):
        self.result['decoder_seven_calls'] = 6
        with self.assertRaises(AssertionError):
            validate_cpu_result(self.result)

    def test_native_call_guard(self):
        source = (ROOT / 'scripts/probe_y62_native_smoke.py').read_text(encoding='utf-8')
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<guard>', 'exec'), namespace)
        check = namespace['check_path_calls']
        check([True] * 7, 7)
        for inputs, count in (([True] * 6, 6), ([True, False] * 7, 7), ([False] * 7, 7)):
            with self.assertRaises(AssertionError):
                check(inputs, count)


if __name__ == '__main__':
    unittest.main()
