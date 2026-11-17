"""Reject altered identities and incomplete actual feedback checks before admission."""
import copy
import unittest
from unittest.mock import patch
import admit_dispatch_y65_gpu0 as handoff


class AdmissionGuards(unittest.TestCase):
    def setUp(self):
        self.c = dict(id='AdaRES-Y65', config='candidate.yaml', code_commit='a' * 40)
        self.f = dict(goal_sha256='b' * 64, code_commit='a' * 40,
            variant=dict(config='candidate.yaml', config_data=dict(sha256='c' * 64)))
        gradient = dict(tensors=540, all_finite=True, indicator_tensors=10, indicator_calls=7,
            feedback_calls=6, feedback_consumed_by_refinement=6,
            query_norm_input_requires_grad=[True] * 7,
            query_mask_and_feedback_input_gradients_finite=True, feedback_direction_gradient_max=.001)
        self.raw = dict(complete=True, id='AdaRES-Y65', config='candidate.yaml', code_commit='a' * 40,
            goal_sha256='b' * 64, native_YAML_sha256='c' * 64, steps=2, optimizer_steps=2, losses=[7., 8.],
            native_entry='tools/train.py::train', split='train', native_data_loaded=True,
            backward_finite=True, intended_path_observed=True, validation_used=False, model_weights_saved=False,
            shared_head_calls=14, feedback_calls=12, query_norm_input_requires_grad=[True] * 14,
            gradient_checks=[copy.deepcopy(gradient), copy.deepcopy(gradient)],
            batches=[dict(scan_ids=['a','b'], expressions=8, points=100), dict(scan_ids=['c','d'], expressions=8, points=101)],
            feedback_paths=[dict(direction_nonzero=i >= 6, centered_max=.2,
                query_delta_max=.01 if i >= 6 else 0., consumed_by_refinement=True) for i in range(12)],
            peak_reserved_bytes=1024, data_source=dict(config_data_root='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/',
                dataset_class='ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'))

    def verify(self):
        handoff.validate_result(self.raw, self.c, self.f)

    def test_valid_fixture_schema_only(self):
        self.verify()

    def test_wrong_id_config_commit_goal_yaml_refused(self):
        for key in ('id','config','code_commit','goal_sha256','native_YAML_sha256'):
            old=copy.deepcopy(self.raw);self.raw[key]='wrong'
            with self.subTest(key=key), self.assertRaises(AssertionError): self.verify()
            self.raw=old

    def test_missing_step_or_nonfinite_loss_refused(self):
        for field,value in (('steps',1),('optimizer_steps',1),('losses',[7.]),('losses',[7.,float('nan')])):
            old=copy.deepcopy(self.raw);self.raw[field]=value
            with self.assertRaises(AssertionError): self.verify()
            self.raw=old

    def test_unexecuted_native_path_refused(self):
        for key in ('native_data_loaded','backward_finite','intended_path_observed'):
            self.raw[key]=False
            with self.assertRaises(AssertionError): self.verify()
            self.raw[key]=True

    def test_validation_or_saved_weights_refused(self):
        for key,value in (('split','val'),('validation_used',True),('model_weights_saved',True)):
            old=copy.deepcopy(self.raw);self.raw[key]=value
            with self.assertRaises(AssertionError): self.verify()
            self.raw=old

    def test_missing_zero_or_nonfinite_feedback_gradient_refused(self):
        for value in (0.,float('inf'),float('nan')):
            self.raw['gradient_checks'][0]['feedback_direction_gradient_max']=value
            with self.assertRaises(AssertionError): self.verify()
        self.raw['gradient_checks'][0].pop('feedback_direction_gradient_max')
        with self.assertRaises(KeyError): self.verify()

    def test_missing_feedback_indicator_or_gradient_stage_refused(self):
        for key,value in (('feedback_calls',5),('indicator_calls',6),('indicator_tensors',9),
                          ('query_norm_input_requires_grad',[False]*7),('query_mask_and_feedback_input_gradients_finite',False)):
            old=copy.deepcopy(self.raw);self.raw['gradient_checks'][0][key]=value
            with self.assertRaises(AssertionError): self.verify()
            self.raw=old

    def test_wrong_anchor_or_missing_second_response_refused(self):
        self.raw['feedback_paths'][0]['query_delta_max']=.1
        with self.assertRaises(AssertionError): self.verify()
        self.raw['feedback_paths'][0]['query_delta_max']=0.
        for row in self.raw['feedback_paths'][6:]: row['query_delta_max']=0.
        with self.assertRaises(AssertionError): self.verify()

    def test_bypassed_or_missing_consumption_refused(self):
        for value in (False,1):
            self.raw['feedback_paths'][6]['consumed_by_refinement']=value
            with self.assertRaises(AssertionError): self.verify()
        self.raw['feedback_paths'][6]['consumed_by_refinement']=True
        self.raw['gradient_checks'][0]['feedback_consumed_by_refinement']=5
        with self.assertRaises(AssertionError): self.verify()

    def test_wrong_data_or_memory_refused(self):
        for key,value in (('config_data_root','/other'),('dataset_class','synthetic')):
            old=copy.deepcopy(self.raw);self.raw['data_source'][key]=value
            with self.assertRaises(AssertionError): self.verify()
            self.raw=old
        self.raw['peak_reserved_bytes']=24*2**30
        with self.assertRaises(AssertionError): self.verify()

    def test_live_idle_only_without_compute_owner(self):
        obs=dict(gpu_query_rc=0,compute_query_rc=0,
            authorized_device_inventory='0, GPU-test, RTX3090, 0, 1, 24576\n',compute='')
        with patch.object(handoff,'snapshot',return_value=obs), patch.object(handoff.monitor,'ssh',return_value='280.0'):
            self.assertEqual(handoff.require_idle(dict(host='test-host',user='test-user',python='/test/python'))['compute'],'')
        for changed in (dict(obs,compute='GPU-test, 123, python, 100\n'),
                        dict(obs,authorized_device_inventory='0, GPU-test, RTX3090, 10, 1, 24576\n'),
                        dict(obs,authorized_device_inventory='0, GPU-test, RTX3090, 0, 1024, 24576\n')):
            with patch.object(handoff,'snapshot',return_value=changed), self.assertRaises(AssertionError):
                handoff.require_idle(dict(host='test-host',user='test-user',python='/test/python'))

    def test_wrong_power_or_failed_query_refused(self):
        obs=dict(gpu_query_rc=0,compute_query_rc=0,
            authorized_device_inventory='0, GPU-test, RTX3090, 0, 1, 24576\n',compute='')
        for power in ('350','nan'):
            with patch.object(handoff,'snapshot',return_value=obs), patch.object(handoff.monitor,'ssh',return_value=power), self.assertRaises(AssertionError):
                handoff.require_idle(dict(host='test-host',user='test-user',python='/test/python'))
        with patch.object(handoff,'snapshot',return_value=dict(obs,gpu_query_rc=1)), self.assertRaises(AssertionError):
            handoff.require_idle(dict(host='test-host',user='test-user',python='/test/python'))


if __name__ == '__main__':
    unittest.main()
