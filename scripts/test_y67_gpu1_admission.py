"""Actual admission requires a complete, identity-bound native result; fixtures prove schema only."""
import copy
import unittest
from unittest.mock import patch
import admit_dispatch_y67_gpu1 as handoff


class AdmissionGuards(unittest.TestCase):
    def setUp(self):
        self.c = dict(id='AdaRES-Y67', config='candidate.yaml', code_commit='a' * 40)
        self.f = dict(goal_sha256='b' * 64, code_commit='a' * 40,
            variant=dict(config='candidate.yaml', config_data=dict(sha256='c' * 64)))
        stages = [dict(stage=i, parameter_tensors=2, all_finite=True,
            gradient_max=.001, parameter_update_max=.0001) for i in range(7)]
        gradient = dict(tensors=550, all_finite=True, native_mask_and_score_consumption=True,
            shared_indicator_calls=7, normalization_stages=stages)
        self.raw = dict(complete=True, id='AdaRES-Y67', config='candidate.yaml', code_commit='a' * 40,
            goal_sha256='b' * 64, native_YAML_sha256='c' * 64, steps=2, optimizer_steps=2, losses=[7., 8.],
            native_entry='tools/train.py::train', split='train', native_data_loaded=True,
            backward_finite=True, intended_path_observed=True, validation_used=False, model_weights_saved=False,
            stage_norm_calls=list(range(7)) * 2, native_mask_calls=14, native_score_calls=14,
            shared_indicator_calls=14, formula_errors=[1e-7] * 14,
            gradient_checks=[copy.deepcopy(gradient), copy.deepcopy(gradient)],
            batches=[dict(scan_ids=['a', 'b'], expressions=8, points=100),
                     dict(scan_ids=['c', 'd'], expressions=8, points=101)],
            peak_reserved_bytes=1024, data_source=dict(
                config_data_root='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/',
                dataset_class='ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'))

    def verify(self):
        handoff.validate_result(self.raw, self.c, self.f)

    def test_valid_fixture_schema_only(self): self.verify()

    def test_wrong_id_config_commit_goal_yaml_refused(self):
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

    def test_missing_reordered_duplicate_or_boolean_stage_refused(self):
        for calls in (list(range(7)), list(reversed(range(7))) * 2, [0] * 14,
                      [False, 1, 2, 3, 4, 5, 6] + list(range(7))):
            self.raw['stage_norm_calls'] = calls
            with self.assertRaises(AssertionError): self.verify()

    def test_missing_native_mask_score_or_indicator_consumption_refused(self):
        for key in ('native_mask_calls', 'native_score_calls', 'shared_indicator_calls'):
            old = copy.deepcopy(self.raw); self.raw[key] = 13
            with self.assertRaises(AssertionError): self.verify()
            self.raw = old

    def test_missing_or_nonfinite_formula_observation_refused(self):
        for values in ([0.] * 13, [float('inf')] * 14, [float('nan')] * 14, [-.1] * 14):
            self.raw['formula_errors'] = values
            with self.assertRaises(AssertionError): self.verify()

    def test_missing_affine_gradient_or_stage_refused(self):
        for change in ('missing', 'tensor', 'stage'):
            old = copy.deepcopy(self.raw)
            rows = self.raw['gradient_checks'][0]['normalization_stages']
            if change == 'missing': rows.pop()
            elif change == 'tensor': rows[6]['parameter_tensors'] = 1
            else: rows[6]['stage'] = 5
            with self.assertRaises(AssertionError): self.verify()
            self.raw = old

    def test_zero_or_nonfinite_gradient_or_actual_update_refused(self):
        for key in ('gradient_max', 'parameter_update_max'):
            for value in (0., float('inf'), float('nan')):
                old = copy.deepcopy(self.raw)
                self.raw['gradient_checks'][0]['normalization_stages'][6][key] = value
                with self.assertRaises(AssertionError): self.verify()
                self.raw = old

    def test_wrong_data_distinct_scenes_or_memory_refused(self):
        for key, value in (('config_data_root', '/other'), ('dataset_class', 'synthetic')):
            old = copy.deepcopy(self.raw); self.raw['data_source'][key] = value
            with self.assertRaises(AssertionError): self.verify()
            self.raw = old
        old = copy.deepcopy(self.raw); self.raw['batches'][1]['scan_ids'] = ['a', 'b']
        with self.assertRaises(AssertionError): self.verify()
        self.raw = old; self.raw['peak_reserved_bytes'] = 24 * 2**30
        with self.assertRaises(AssertionError): self.verify()

    def test_failed_or_bypassed_per_step_path_refused(self):
        for key, value in (('all_finite', False), ('native_mask_and_score_consumption', False),
                           ('native_mask_and_score_consumption', 1), ('shared_indicator_calls', 6)):
            old = copy.deepcopy(self.raw); self.raw['gradient_checks'][0][key] = value
            with self.assertRaises(AssertionError): self.verify()
            self.raw = old

    def test_live_idle_only_without_compute_owner_or_busy_device(self):
        obs = dict(gpu_query_rc=0, compute_query_rc=0,
            authorized_device_inventory='1, GPU-test, RTX3090, 0, 1, 24576\n', compute='')
        server = dict(host='test-host', user='test-user', python='/test/python')
        with patch.object(handoff, 'snapshot', return_value=obs), patch.object(handoff.monitor, 'ssh', return_value='390.0'):
            self.assertEqual(handoff.require_idle(server)['compute'], '')
        for changed in (dict(obs, compute='GPU-test, 123, python, 100\n'),
                        dict(obs, authorized_device_inventory='1, GPU-test, RTX3090, 10, 1, 24576\n'),
                        dict(obs, authorized_device_inventory='1, GPU-test, RTX3090, 0, 1024, 24576\n')):
            with patch.object(handoff, 'snapshot', return_value=changed), self.assertRaises(AssertionError):
                handoff.require_idle(server)

    def test_wrong_power_or_failed_query_refused(self):
        obs = dict(gpu_query_rc=0, compute_query_rc=0,
            authorized_device_inventory='1, GPU-test, RTX3090, 0, 1, 24576\n', compute='')
        server = dict(host='test-host', user='test-user', python='/test/python')
        for power in ('280', '350', 'nan'):
            with patch.object(handoff, 'snapshot', return_value=obs), patch.object(handoff.monitor, 'ssh', return_value=power), self.assertRaises(AssertionError):
                handoff.require_idle(server)
        with patch.object(handoff, 'snapshot', return_value=dict(obs, gpu_query_rc=1)), self.assertRaises(AssertionError):
            handoff.require_idle(server)


if __name__ == '__main__': unittest.main()
