"""Negative admission fixtures are not native TRAIN evidence."""
import copy
import unittest
import admit_dispatch_y69_server1 as handoff


class AdmissionGuards(unittest.TestCase):
    def setUp(self):
        self.c=dict(id='AdaRES-Y69',config='candidate.yaml',code_commit='a'*40)
        self.f=dict(goal_sha256='b'*64,variant=dict(config='candidate.yaml',config_data=dict(sha256='c'*64)),code_commit='a'*40)
        g=dict(tensors=540,all_finite=True,indicator_tensors=10,indicator_calls=7,geometry_calls=1,
            coarse_stage_calls=list(range(6)),attention_calls=6,native_hard_masks_preserved=True,
            source_and_query_gradients_finite=True,geometry_weight_gradient_max=.001,parameter_update_max=.0001)
        self.raw=dict(complete=True,id='AdaRES-Y69',config='candidate.yaml',code_commit='a'*40,
            goal_sha256='b'*64,native_YAML_sha256='c'*64,steps=2,optimizer_steps=2,losses=[7.,8.],
            native_entry='tools/train.py::train',split='train',native_data_loaded=True,backward_finite=True,
            intended_path_observed=True,validation_used=False,model_weights_saved=False,
            shared_head_calls=14,geometry_calls=2,coarse_stage_calls=list(range(6))*2,
            attention_stage_calls=list(range(6))*2,gradient_checks=[copy.deepcopy(g),copy.deepcopy(g)],
            geometry_paths=[dict(weight_nonzero=bool(i),bias_max=.01 if i else 0.,reference_error_max=1e-7,
                native_positions_verified=True,hard_masks_verified=True,consumed_by_all_coarse_attentions=True,
                expressions=8,heads=8,queries=128,reference_expression_rows=[0,7],reference_head_rows=[0,7],
                reference_query_rows=[0,127]) for i in range(2)],
            batches=[dict(scan_ids=['a','b'],expressions=8,points=100),dict(scan_ids=['c','d'],expressions=8,points=101)],
            peak_reserved_bytes=1024,data_source=dict(config_data_root='/data1/songcl/Project/AdaRES_project/data/',
                dataset_class='ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'))

    def verify(self):handoff.validate_result(self.raw,self.c,self.f)

    def test_valid_fixture_schema_only(self):self.verify()

    def test_wrong_identity_or_yaml(self):
        for key in ('id','config','code_commit','goal_sha256','native_YAML_sha256'):
            old=copy.deepcopy(self.raw);self.raw[key]='wrong'
            with self.assertRaises(AssertionError):self.verify()
            self.raw=old

    def test_steps_or_finite_losses(self):
        for k,v in (('steps',1),('optimizer_steps',1),('losses',[7.]),('losses',[7.,float('nan')])):
            old=copy.deepcopy(self.raw);self.raw[k]=v
            with self.assertRaises(AssertionError):self.verify()
            self.raw=old

    def test_validation_or_weights_saved(self):
        for k,v in (('split','val'),('validation_used',True),('model_weights_saved',True)):
            old=copy.deepcopy(self.raw);self.raw[k]=v
            with self.assertRaises(AssertionError):self.verify()
            self.raw=old

    def test_nonboolean_or_unexecuted_path(self):
        for k in ('native_data_loaded','backward_finite','intended_path_observed'):
            for v in (False,1):
                old=copy.deepcopy(self.raw);self.raw[k]=v
                with self.assertRaises(AssertionError):self.verify()
                self.raw=old

    def test_missing_reordered_or_boolean_stage(self):
        for k,v in (('coarse_stage_calls',list(range(6))),('attention_stage_calls',[0]*12),
                    ('shared_head_calls',13),('geometry_calls',1),('coarse_stage_calls',[False,1,2,3,4,5]*2)):
            old=copy.deepcopy(self.raw);self.raw[k]=v
            with self.assertRaises(AssertionError):self.verify()
            self.raw=old

    def test_gradient_path_or_native_hard_exclusion(self):
        for k,v in (('all_finite',False),('indicator_tensors',9),('indicator_calls',6),('geometry_calls',0),
                    ('attention_calls',5),('coarse_stage_calls',[0]*6),('native_hard_masks_preserved',False),
                    ('source_and_query_gradients_finite',False)):
            old=copy.deepcopy(self.raw);self.raw['gradient_checks'][0][k]=v
            with self.assertRaises(AssertionError):self.verify()
            self.raw=old

    def test_new_weight_gradient_or_update(self):
        for k in ('geometry_weight_gradient_max','parameter_update_max'):
            for v in (0.,float('inf'),float('nan')):
                old=copy.deepcopy(self.raw);self.raw['gradient_checks'][0][k]=v
                with self.assertRaises(AssertionError):self.verify()
                self.raw=old

    def test_initial_anchor_and_second_response(self):
        for i,k,v in ((0,'weight_nonzero',True),(0,'bias_max',.1),(1,'weight_nonzero',False),(1,'bias_max',0.)):
            old=copy.deepcopy(self.raw);self.raw['geometry_paths'][i][k]=v
            with self.assertRaises(AssertionError):self.verify()
            self.raw=old

    def test_wrong_or_bypassed_geometry_consumption(self):
        for k in ('native_positions_verified','hard_masks_verified','consumed_by_all_coarse_attentions'):
            for v in (False,1):
                old=copy.deepcopy(self.raw);self.raw['geometry_paths'][1][k]=v
                with self.assertRaises(AssertionError):self.verify()
                self.raw=old

    def test_wrong_reference_rows_or_nonfinite_formula(self):
        for k,v in (('reference_head_rows',[1,7]),('reference_query_rows',[0,126]),
                    ('reference_expression_rows',[0,6]),('heads',1),('reference_error_max',float('nan'))):
            old=copy.deepcopy(self.raw);self.raw['geometry_paths'][1][k]=v
            with self.assertRaises(AssertionError):self.verify()
            self.raw=old

    def test_wrong_data_or_memory(self):
        for k,v in (('config_data_root','/other'),('dataset_class','synthetic')):
            old=copy.deepcopy(self.raw);self.raw['data_source'][k]=v
            with self.assertRaises(AssertionError):self.verify()
            self.raw=old
        self.raw['peak_reserved_bytes']=24*2**30
        with self.assertRaises(AssertionError):self.verify()

    def test_actual_idle_and_busy_compute_occupation(self):
        self.assertEqual(handoff.validate_idle(dict(gpu='0, 98, 350',compute='')),350.)
        for value in (dict(gpu='0, 98, 350',compute='123'),dict(gpu='10, 98, 350',compute=''),dict(gpu='0, 1024, 350',compute='')):
            with self.assertRaises(AssertionError):handoff.validate_idle(value)

    def test_wrong_power_or_nonfinite_telemetry(self):
        for value in ('0,98,280','nan,98,350','-1,98,350','0,inf,350'):
            with self.assertRaises(AssertionError):handoff.validate_idle(dict(gpu=value,compute=''))


if __name__=='__main__':unittest.main()
