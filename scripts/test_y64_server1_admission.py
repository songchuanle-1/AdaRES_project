"""Negative admission fixtures; actual admission requires bound native results."""
import copy
import unittest
import admit_dispatch_y64_server1 as handoff


class AdmissionGuards(unittest.TestCase):
    def setUp(self):
        self.candidate = dict(id='AdaRES-Y64', config='candidate.yaml', code_commit='a' * 40)
        self.frozen = dict(goal_sha256='b' * 64, variant=dict(config_data=dict(sha256='c' * 64)))
        gradient = dict(all_finite=True, indicator_tensors=11, indicator_calls=7,
            query_norm_input_requires_grad=[True] * 7, query_and_mask_input_gradients_finite=True,
            trainable_query_norm_gradient_nonzero=True, containment_weight_gradient_max=.001)
        self.raw = dict(complete=True, id='AdaRES-Y64', config='candidate.yaml', code_commit='a' * 40,
            goal_sha256='b' * 64, native_YAML_sha256='c' * 64, steps=2, optimizer_steps=2, losses=[7., 8.],
            native_entry='tools/train.py::train', split='train', native_data_loaded=True,
            backward_finite=True, intended_path_observed=True, validation_used=False, model_weights_saved=False,
            shared_head_calls=14, query_norm_input_requires_grad=[True] * 14,
            gradient_checks=[copy.deepcopy(gradient), copy.deepcopy(gradient)],
            batches=[dict(scan_ids=['a', 'b'], expressions=8, points=100)] * 2,
            containment_paths=[dict(containment_weight_nonzero=i >= 7, context_abs_max=1.,
                logit_delta_max=.01 if i >= 7 else 0.) for i in range(14)], peak_reserved_bytes=1024,
            data_source=dict(config_data_root='/data1/songcl/Project/AdaRES_project/data/'))

    def verify(self):
        handoff.validate_result(self.raw, self.candidate, self.frozen)

    def test_fixture_schema_only(self):
        self.verify()

    def test_wrong_id_config_commit_goal_yaml_refused(self):
        for key in ('id', 'config', 'code_commit', 'goal_sha256', 'native_YAML_sha256'):
            old = copy.deepcopy(self.raw)
            self.raw[key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(AssertionError): self.verify()
            self.raw = old

    def test_missing_second_step_refused(self):
        self.raw['steps'] = 1
        with self.assertRaises(AssertionError): self.verify()

    def test_nonfinite_loss_refused(self):
        self.raw['losses'][0] = float('nan')
        with self.assertRaises(AssertionError): self.verify()

    def test_unexecuted_native_path_refused(self):
        for key in ('native_data_loaded', 'backward_finite', 'intended_path_observed'):
            self.raw[key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError): self.verify()
            self.raw[key] = True

    def test_validation_or_saved_weights_refused(self):
        self.raw['split'] = 'val'
        with self.assertRaises(AssertionError): self.verify()
        self.raw['split'] = 'train'
        self.raw['model_weights_saved'] = True
        with self.assertRaises(AssertionError): self.verify()

    def test_zero_nonfinite_or_missing_containment_gradient_refused(self):
        for value in (0., float('inf')):
            self.raw['gradient_checks'][0]['containment_weight_gradient_max'] = value
            with self.subTest(value=value), self.assertRaises(AssertionError): self.verify()

    def test_initial_or_second_response_refused(self):
        self.raw['containment_paths'][0]['logit_delta_max'] = .1
        with self.assertRaises(AssertionError): self.verify()
        self.raw['containment_paths'][0]['logit_delta_max'] = 0.
        for r in self.raw['containment_paths'][7:]: r['logit_delta_max'] = 0.
        with self.assertRaises(AssertionError): self.verify()

    def test_wrong_data_or_memory_bound_refused(self):
        self.raw['data_source']['config_data_root'] = '/other'
        with self.assertRaises(AssertionError): self.verify()
        self.raw['data_source']['config_data_root'] = '/data1/songcl/Project/AdaRES_project/data'
        self.raw['peak_reserved_bytes'] = 24 * 2**30
        with self.assertRaises(AssertionError): self.verify()

    def test_actual_idle_schema_only(self):
        self.assertEqual(handoff.validate_idle(dict(gpu='0, 98, 350', compute='')), 350.)

    def test_compute_pid_refused(self):
        with self.assertRaises(AssertionError):
            handoff.validate_idle(dict(gpu='0, 98, 350', compute='123'))

    def test_busy_gpu_memory_or_utilization_refused(self):
        for gpu in ('10, 98, 350', '0, 1024, 350'):
            with self.subTest(gpu=gpu), self.assertRaises(AssertionError):
                handoff.validate_idle(dict(gpu=gpu, compute=''))

    def test_wrong_power_nonfinite_or_negative_refused(self):
        for gpu in ('0, 98, 280', 'nan, 98, 350', '-1, 98, 350'):
            with self.subTest(gpu=gpu), self.assertRaises(AssertionError):
                handoff.validate_idle(dict(gpu=gpu, compute=''))


if __name__ == '__main__':
    unittest.main()
