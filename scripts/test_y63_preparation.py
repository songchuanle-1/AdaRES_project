"""CPU guard metadata tests; real probes remain mandatory."""
import ast
import unittest
from prepare_y63_peer_bilinear import ROOT, validate_cpu_result

class PreparationTests(unittest.TestCase):
    def setUp(self):
        controls = [dict(dtype=dtype, zero_anchor_output_and_input_gradients_exact=True,
            finite_parameter_gradients=True, bilinear_weight_gradient_nonzero=True,
            permutation_padding_singleton_empty_controls=True, initial_RNG_matches_parent=True,
            explicit_cross_channel_formula_matched=True, transposed_axis_negative_control_detected=True,
            nonzero_residual_path_observed=True) for dtype in ('torch.float32', 'torch.float64')]
        self.result = dict(complete=True, id='AdaRES-Y63', parameters=164760709,
            additional_parameters=131072, decoder_seven_calls=7, component_controls=controls)
        for key in ('implementation_test_passed', 'full_model_CPU_constructed',
                    'configured_TRAIN_data_collated', 'released_backbone_loaded',
                    'native_YAML_semantically_unchanged', 'full_decoder_CPU_executed',
                    'decoder_anchor_all_stages_exact', 'decoder_backward_finite',
                    'decoder_bilinear_gradient_nonzero'):
            self.result[key] = True
        for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward',
                    'native_sparse_forward', 'validation_used', 'admission'):
            self.result[key] = False

    def test_valid_metadata(self):
        validate_cpu_result(self.result)

    def test_missing_dtype(self):
        self.result['component_controls'].pop()
        with self.assertRaises(AssertionError):
            validate_cpu_result(self.result)

    def test_wrong_bilinear_axes_refused(self):
        self.result['component_controls'][0]['transposed_axis_negative_control_detected'] = False
        with self.assertRaises(AssertionError):
            validate_cpu_result(self.result)

    def test_wrong_parameter_budget(self):
        self.result['parameters'] += 1
        with self.assertRaises(AssertionError):
            validate_cpu_result(self.result)

    def test_CPU_is_not_native(self):
        self.result['native_smoke_passed'] = True
        with self.assertRaises(AssertionError):
            validate_cpu_result(self.result)

    def test_seven_stages_required(self):
        self.result['decoder_seven_calls'] = 6
        with self.assertRaises(AssertionError):
            validate_cpu_result(self.result)

    def test_extra_RNG_refused(self):
        self.result['component_controls'][1]['initial_RNG_matches_parent'] = False
        with self.assertRaises(AssertionError):
            validate_cpu_result(self.result)

    def test_native_call_guard(self):
        source = (ROOT / 'scripts/probe_y63_native_smoke.py').read_text(encoding='utf-8')
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
