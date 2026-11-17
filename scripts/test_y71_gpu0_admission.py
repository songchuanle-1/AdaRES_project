"""Reject incomplete native evidence and unsafe keeper/source mutations."""
import copy
import unittest
from admit_dispatch_y71_gpu0 import validate_result,validate_additive_delta,validate_keeper

class Admission(unittest.TestCase):
    def setUp(self):
        self.c=dict(id='AdaRES-Y71',config='candidate.yaml',code_commit='a'*40)
        self.f=dict(code_commit='a'*40,goal_sha256='b'*64,variant=dict(config='candidate.yaml',config_data=dict(sha256='c'*64)))
        paths=[dict(weight_nonzero=bool(i),native_projection_identity_verified=True,normalization_consumption_verified=True,
            feature_rows=100,reference_feature_rows=[0,50,99],reference_channels=[0,127,255],reference_error_max=1e-7,
            fusion_delta_max=.01 if i else 0.,spatial_abs_max=3.,appearance_abs_max=4.) for i in range(2)]
        grad=dict(all_finite=True,tensors=540,indicator_tensors=10,indicator_calls=7,fusion_calls=1,
            normalization_consumption_calls=1,projected_branch_gradients_finite=True,
            fusion_weight_gradient_max=1e-5,parameter_update_max=1e-4)
        self.raw=dict(complete=True,id='AdaRES-Y71',config='candidate.yaml',code_commit='a'*40,goal_sha256='b'*64,
            native_YAML_sha256='c'*64,steps=2,optimizer_steps=2,losses=[7.8,8.2],native_entry='tools/train.py::train',split='train',
            native_data_loaded=True,backward_finite=True,intended_path_observed=True,native_modes_verified=True,
            validation_used=False,model_weights_saved=False,fusion_calls=2,shared_head_calls=14,
            normalization_consumption_calls=2,fusion_paths=paths,gradient_checks=[copy.deepcopy(grad),copy.deepcopy(grad)],
            batches=[dict(scan_ids=['a','b'],expressions=8,points=310000),dict(scan_ids=['c','d'],expressions=8,points=320000)],
            peak_reserved_bytes=7*2**30,data_source=dict(dataset_class='ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge',config_data_root='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/'))
    def verify(self,raw=None):validate_result(self.raw if raw is None else raw,self.c,self.f)
    def test_valid_fixture_schema_only(self):self.verify()
    def test_wrong_id_config_commit_goal_or_yaml(self):
        for key,value in (('id','other'),('config','other.yaml'),('code_commit','d'*40),('goal_sha256','d'*64),('native_YAML_sha256','d'*64)):
            with self.subTest(key=key),self.assertRaises(AssertionError):self.verify(dict(self.raw,**{key:value}))
    def test_wrong_native_entry_split_or_step_count(self):
        for key,value in (('steps',1),('optimizer_steps',3),('native_entry','fixture'),('split','val')):
            with self.assertRaises(AssertionError):self.verify(dict(self.raw,**{key:value}))
    def test_nonfinite_or_incomplete_losses(self):
        for value in ([7.8],[float('nan'),8.2],[7.8,float('inf')]):
            with self.assertRaises(AssertionError):self.verify(dict(self.raw,losses=value))
    def test_unloaded_detached_unobserved_or_wrong_modes(self):
        for key in ('native_data_loaded','backward_finite','intended_path_observed','native_modes_verified'):
            for value in (False,1):
                with self.assertRaises(AssertionError):self.verify(dict(self.raw,**{key:value}))
    def test_unexecuted_or_reordered_path_counts(self):
        for key,value in (('fusion_calls',1),('shared_head_calls',13),('normalization_consumption_calls',1)):
            with self.assertRaises(AssertionError):self.verify(dict(self.raw,**{key:value}))
    def test_missing_or_zero_second_response(self):
        for key,value in (('weight_nonzero',False),('fusion_delta_max',0.)):
            raw=copy.deepcopy(self.raw);raw['fusion_paths'][1][key]=value
            with self.assertRaises(AssertionError):self.verify(raw)
    def test_bypassed_or_wrong_branch_identity(self):
        for key in ('native_projection_identity_verified','normalization_consumption_verified'):
            raw=copy.deepcopy(self.raw);raw['fusion_paths'][1][key]=False
            with self.assertRaises(AssertionError):self.verify(raw)
    def test_nonfinite_zero_or_missing_gradient_update(self):
        for key in ('fusion_weight_gradient_max','parameter_update_max'):
            for value in (0.,-.1,float('nan'),float('inf')):
                raw=copy.deepcopy(self.raw);raw['gradient_checks'][1][key]=value
                with self.assertRaises(AssertionError):self.verify(raw)
    def test_detached_projected_input_refused(self):
        raw=copy.deepcopy(self.raw);raw['gradient_checks'][0]['projected_branch_gradients_finite']=False
        with self.assertRaises(AssertionError):self.verify(raw)
    def test_validation_selection_or_diagnostic_weights(self):
        for key in ('validation_used','model_weights_saved'):
            with self.assertRaises(AssertionError):self.verify(dict(self.raw,**{key:True}))
    def test_memory_budget_or_data_source(self):
        for value in (0,24*2**30):
            with self.assertRaises(AssertionError):self.verify(dict(self.raw,peak_reserved_bytes=value))
        for key in ('dataset_class','config_data_root'):
            raw=copy.deepcopy(self.raw);raw['data_source'][key]='other'
            with self.assertRaises(AssertionError):self.verify(raw)
    def test_only_seven_exact_new_files_accepted(self):
        names=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py');rows=[]
        for package in ('adares_y71_visual_product_fusion',):
            rows.extend([['A','AdaRES/ipdn/model_paper5/'+package+'/'+n] for n in names]);rows.append(['A','AdaRES/configs_song/model_paper5/'+package+'_res.yaml'])
        validate_additive_delta(rows)
        for changed in (rows[:-1],[['M',rows[0][1]]]+rows[1:],[['A','AdaRES/tools/train.py']]+rows[1:]):
            with self.assertRaises(AssertionError):validate_additive_delta(changed)
    def test_existing_gpu1_identity_must_stay_exact(self):
        keeper=dict(id='AdaRES-Y70',status='running',launch_verified=True,gpu='1',pid=52561,
            code_commit='39bdb8d9783512e70cdc2763e9a6097ab0f67a0f',config='configs_song/model_paper5/adares_y70_query_scale_fusion_res.yaml')
        validate_keeper(keeper)
        for key,value in (('gpu','0'),('pid',1),('code_commit','other'),('status','claiming'),('launch_verified',False)):
            with self.assertRaises(AssertionError):validate_keeper(dict(keeper,**{key:value}))

if __name__=='__main__':unittest.main()
