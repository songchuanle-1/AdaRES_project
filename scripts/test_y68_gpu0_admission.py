"""Admission guard fixtures validate rejection logic, never actual training."""
import copy
import unittest
from unittest.mock import patch

import admit_dispatch_y68_gpu0 as handoff


class AdmissionGuards(unittest.TestCase):
    def setUp(self):
        self.c = dict(id='AdaRES-Y68', config='candidate.yaml', code_commit='a' * 40)
        self.f = dict(goal_sha256='b' * 64, code_commit='a' * 40,
            variant=dict(config='candidate.yaml', config_data=dict(sha256='c' * 64)))
        gradient = dict(tensors=540, all_finite=True, indicator_tensors=10, indicator_calls=7,
            query_norm_input_requires_grad=[True] * 7, modulation_calls=1,
            prediction_consumption_calls=7, feature_and_language_gradients_finite=True,
            modulation_weight_gradient_max=.001, parameter_update_max=.0001)
        self.raw = dict(complete=True, id='AdaRES-Y68', config='candidate.yaml', code_commit='a' * 40,
            goal_sha256='b' * 64, native_YAML_sha256='c' * 64, steps=2, optimizer_steps=2, losses=[7., 8.],
            native_entry='tools/train.py::train', split='train', native_data_loaded=True,
            backward_finite=True, intended_path_observed=True, validation_used=False, model_weights_saved=False,
            shared_head_calls=14, modulation_calls=2, prediction_consumption_calls=14,
            query_norm_input_requires_grad=[True] * 14,
            gradient_checks=[copy.deepcopy(gradient), copy.deepcopy(gradient)],
            modulation_paths=[dict(projection_nonzero=bool(i), sentence_max=.2,
                feature_delta_max=.01 if i else 0., reference_error_max=1e-7,
                reference_expression_rows=[0, 7], expressions=8, valid_token_counts=[7] * 8,
                native_input_correspondence_verified=True, consumed_by_all_predictions=True) for i in range(2)],
            batches=[dict(scan_ids=['a', 'b'], expressions=8, points=100),
                     dict(scan_ids=['c', 'd'], expressions=8, points=101)],
            peak_reserved_bytes=1024, data_source=dict(
                config_data_root='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/',
                dataset_class='ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'))

    def verify(self):
        handoff.validate_result(self.raw, self.c, self.f)

    def test_valid_fixture_schema_only(self):
        self.verify()

    def test_wrong_identity_or_yaml_refused(self):
        for key in ('id', 'config', 'code_commit', 'goal_sha256', 'native_YAML_sha256'):
            old = copy.deepcopy(self.raw); self.raw[key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(AssertionError): self.verify()
            self.raw = old

    def test_missing_step_or_nonfinite_loss_refused(self):
        for key, value in (('steps', 1), ('optimizer_steps', 1), ('losses', [7.]), ('losses', [7., float('nan')])):
            old = copy.deepcopy(self.raw); self.raw[key] = value
            with self.assertRaises(AssertionError): self.verify()
            self.raw = old

    def test_validation_or_saved_weights_refused(self):
        for key, value in (('split', 'val'), ('validation_used', True), ('model_weights_saved', True)):
            old = copy.deepcopy(self.raw); self.raw[key] = value
            with self.assertRaises(AssertionError): self.verify()
            self.raw = old

    def test_unexecuted_or_nonboolean_native_path_refused(self):
        for key in ('native_data_loaded', 'backward_finite', 'intended_path_observed'):
            for value in (False, 1):
                old = copy.deepcopy(self.raw); self.raw[key] = value
                with self.assertRaises(AssertionError): self.verify()
                self.raw = old

    def test_missing_prediction_modulation_or_shared_head_refused(self):
        for key, value in (('shared_head_calls', 13), ('prediction_consumption_calls', 13),
                           ('modulation_calls', 1), ('query_norm_input_requires_grad', [1] * 14)):
            old = copy.deepcopy(self.raw); self.raw[key] = value
            with self.assertRaises(AssertionError): self.verify()
            self.raw = old

    def test_missing_or_detached_gradient_path_refused(self):
        for key, value in (('indicator_tensors', 9), ('indicator_calls', 6), ('modulation_calls', 0),
                           ('prediction_consumption_calls', 6), ('query_norm_input_requires_grad', [False] * 7),
                           ('feature_and_language_gradients_finite', False), ('all_finite', False)):
            old = copy.deepcopy(self.raw); self.raw['gradient_checks'][0][key] = value
            with self.assertRaises(AssertionError): self.verify()
            self.raw = old

    def test_zero_or_nonfinite_new_gradient_and_update_refused(self):
        for key in ('modulation_weight_gradient_max', 'parameter_update_max'):
            for value in (0., float('inf'), float('nan')):
                old = copy.deepcopy(self.raw); self.raw['gradient_checks'][0][key] = value
                with self.assertRaises(AssertionError): self.verify()
                self.raw = old

    def test_wrong_initial_anchor_or_missing_second_response_refused(self):
        for index, key, value in ((0, 'projection_nonzero', True), (0, 'feature_delta_max', .1),
                                  (1, 'projection_nonzero', False), (1, 'feature_delta_max', 0.)):
            old = copy.deepcopy(self.raw); self.raw['modulation_paths'][index][key] = value
            with self.assertRaises(AssertionError): self.verify()
            self.raw = old

    def test_wrong_native_input_or_bypassed_consumption_refused(self):
        for key in ('native_input_correspondence_verified', 'consumed_by_all_predictions'):
            for value in (False, 1):
                old = copy.deepcopy(self.raw); self.raw['modulation_paths'][1][key] = value
                with self.assertRaises(AssertionError): self.verify()
                self.raw = old

    def test_invalid_reference_rows_or_padding_counts_refused(self):
        for key, value in (('reference_expression_rows', [0, 6]), ('valid_token_counts', [0] * 8),
                           ('reference_error_max', float('nan'))):
            old = copy.deepcopy(self.raw); self.raw['modulation_paths'][1][key] = value
            with self.assertRaises(AssertionError): self.verify()
            self.raw = old

    def test_wrong_data_repeated_scene_or_memory_refused(self):
        for key, value in (('config_data_root', '/other'), ('dataset_class', 'synthetic')):
            old = copy.deepcopy(self.raw); self.raw['data_source'][key] = value
            with self.assertRaises(AssertionError): self.verify()
            self.raw = old
        old = copy.deepcopy(self.raw); self.raw['batches'][1]['scan_ids'] = ['a', 'b']
        with self.assertRaises(AssertionError): self.verify()
        self.raw = old; self.raw['peak_reserved_bytes'] = 24 * 2**30
        with self.assertRaises(AssertionError): self.verify()

    def test_idle_requires_no_compute_owner_or_high_occupation(self):
        obs = dict(gpu_query_rc=0, compute_query_rc=0,
            authorized_device_inventory='0, GPU-test, RTX3090, 0, 1, 24576\n', compute='')
        server = dict(host='test-host', user='test-user', python='/test/python')
        with patch.object(handoff, 'snapshot', return_value=obs), patch.object(handoff.monitor, 'ssh', return_value='280.0'):
            self.assertEqual(handoff.require_idle(server)['compute'], '')
        for changed in (dict(obs, compute='GPU-test, 123\n'),
                        dict(obs, authorized_device_inventory='0, GPU-test, RTX3090, 10, 1, 24576\n'),
                        dict(obs, authorized_device_inventory='0, GPU-test, RTX3090, 0, 1024, 24576\n')):
            with patch.object(handoff, 'snapshot', return_value=changed), self.assertRaises(AssertionError):
                handoff.require_idle(server)

    def test_wrong_power_or_failed_query_refused(self):
        obs = dict(gpu_query_rc=0, compute_query_rc=0,
            authorized_device_inventory='0, GPU-test, RTX3090, 0, 1, 24576\n', compute='')
        server = dict(host='test-host', user='test-user', python='/test/python')
        for power in ('390', 'nan'):
            with patch.object(handoff, 'snapshot', return_value=obs), patch.object(handoff.monitor, 'ssh', return_value=power), self.assertRaises(AssertionError):
                handoff.require_idle(server)
        with patch.object(handoff, 'snapshot', return_value=dict(obs, gpu_query_rc=1)), self.assertRaises(AssertionError):
            handoff.require_idle(server)


if __name__ == '__main__':
    unittest.main()
