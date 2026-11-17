"""Exercise operational guards against the actual Y102 native observations."""
import ast,copy,unittest
from prepare_y60_stage_heads import ROOT,monitor,ref,tick
from check_ef_execution import validate_frozen
import advance_y102 as advance
import y102_deployment as deploy
class Admission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=tick.load(ROOT/'pipeline/research_work.json');cls.c,cls.plan,_,cls.f=advance.prepared(cls.work)
        cls.raw=tick.load(ROOT/cls.c['native_smoke_result']['path'])
        cls.parentref=next(r for r in cls.work['candidates'] if r['id']=='AdaRES-Y100')['post_training_freeze'];cls.parent=tick.load(ROOT/cls.parentref['path'])
    def check(self,v=None):advance.validate_result(self.raw if v is None else v,self.c,self.f)
    def test_actual_native_and_exact_seven_additions(self):
        self.check();self.assertEqual(validate_frozen(self.c,self.plan,self.f),7)
        self.assertEqual((self.f['runtime_server'],self.f['runtime_power_limit_watts']),('server-2',280.))
        self.assertEqual(deploy.git('rev-parse',deploy.Y102+'^').decode().strip(),deploy.Y101)
        rows=[['A',p] for p in deploy.paths('AdaRES-Y102')];deploy.validate_delta(rows,'AdaRES-Y102')
        for bad in (rows[:-1],rows+[['A','foreign.py']],[['M',rows[0][1]]]+rows[1:]):
            with self.assertRaises(AssertionError):deploy.validate_delta(bad,'AdaRES-Y102')
        self.assertEqual(list(deploy.contract(deploy.Y101,deploy.Y102,'AdaRES-Y101')),[deploy.Y101,deploy.Y102])
    def test_identity_goal_version_yaml(self):
        for k,v in (('id','foreign'),('code_commit','0'*40),('config','wrong.yaml'),('goal_sha256','0'*64),('native_YAML_sha256','0'*64)):
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(dict(self.raw,**{k:v}))
    def test_real_train_two_finite_steps(self):
        for k,v in (('split','val'),('native_entry','synthetic'),('steps',1),('optimizer_steps',1),('losses',[1.]),('losses',[1.,float('nan')]),('validation_used',True),('model_weights_saved',True),('admission',True)):
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(dict(self.raw,**{k:v}))
    def test_original_and_new_consumers(self):
        for k in ('complete','native_data_loaded','backward_finite','intended_path_observed','original_base_mask_and_static_reuse_preserved','original_hard_gate_algorithm_preserved','native_modes_verified'):
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(dict(self.raw,**{k:False}))
        for k in ('geometry_mask_calls','original_visual_projection_and_mask_calls','coarse_gate_consumers','shared_head_calls','query_norm_consumption_calls'):
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(dict(self.raw,**{k:0}))
    def test_original_gradients_and_parameter_updates(self):
        for k,v in (('all_finite',False),('original_base_adapted_and_visual_gradients_finite',False),('original_visual_parameter_tensors',0),('geometry_mask_calls',0),('frozen_text_encoder_preserved',False),('new_gradient_max',0.),('parameter_update_max',float('nan'))):
            changed=copy.deepcopy(self.raw);changed['gradient_checks'][1][k]=v
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(changed)
    def test_geometry_mass_batching_and_static_seven(self):
        keys=('original_full3_coordinates_and_point_masses','original_unbatched_mask_and_batching','original_spatial_and_appearance_projections_executed','float64_all_valid_scene_centroid_reference','original_padding_rows_preserved','original_base_undetached','all_seven_predictions_reuse_same_features')
        for k in keys:
            changed=copy.deepcopy(self.raw);changed['geometry_mask_paths'][1][k]=False
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(changed)
        for k,v in (('weight_nonzero',False),('mask_feature_delta_max',0.)):
            changed=copy.deepcopy(self.raw);changed['geometry_mask_paths'][1][k]=v
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(changed)
    def test_exact_data_scene_separation_and_budget(self):
        changed=copy.deepcopy(self.raw);changed['data_source']['config_data_root']='/wrong/data'
        with self.assertRaises(AssertionError):self.check(changed)
        changed=copy.deepcopy(self.raw);changed['batches'][1]=copy.deepcopy(changed['batches'][0])
        with self.assertRaises(AssertionError):self.check(changed)
        for v in (0,24*2**30,float('nan')):
            with self.assertRaises(AssertionError):self.check(dict(self.raw,peak_reserved_bytes=v))
    def test_full_original_server2_protocol(self):
        before=copy.deepcopy(self.parent);frozen=advance.make_freeze(self.parent,self.parentref);self.assertEqual(self.parent,before)
        self.assertEqual(frozen['training'],self.parent['training']);restored=copy.deepcopy(frozen['evaluation']);restored['methods'][0]='AdaRES-Y100'
        self.assertEqual(restored,self.parent['evaluation']);self.assertEqual(frozen['resource_condition'],self.parent['resource_condition']);self.assertEqual(frozen['resource_authorization'],ref(advance.GRANT))
        for k,v in (('id','foreign'),('code_commit','0'*40),('complete',False)):
            with self.assertRaises(AssertionError):advance.make_freeze(dict(self.parent,**{k:v}),self.parentref)
    def test_pauses_runtime_busy_and_reservations(self):
        servers=monitor.load_servers(str(ROOT));self.assertEqual(servers['server-1']['allowed_gpus'],[0]);self.assertEqual(servers['server-2']['allowed_gpus'],[0]);self.assertEqual(servers['server-6']['allowed_gpus'],[]);self.assertEqual(ref(advance.PAUSE)['sha256'],advance.PAUSE_SHA)
        for k,v in (('python','/wrong/python'),('expected_hostname','foreign'),('allowed_gpus',[0,1])):
            with self.assertRaises(AssertionError):advance.runtime('server-2',self.c,self.f,dict(servers['server-2'],**{k:v}))
        advance.idle(dict(gpu='0,1,280',compute=''),280.)
        for gpu,compute in (('nan,0,280',''),('0,0,350',''),('0,2000,280',''),('0,0,280','99')):
            with self.assertRaises(AssertionError):advance.idle(dict(gpu=gpu,compute=compute),280.)
        q,r=monitor.load_queue(str(ROOT)),monitor.load_running(str(ROOT));advance.unreserved(q,r,'server-2',0)
        with self.assertRaises(AssertionError):advance.unreserved(q,r,'server-1',0)
    def test_original_full_checkpoint_and_actual_native_handoff(self):
        advance.require_completed_owner();deploy.require_running_parent_handoff(self.work,self.c)
        changed=copy.deepcopy(self.c);changed['native_smoke_result']=dict(path='missing.json',sha256='0'*64)
        with self.assertRaises(AssertionError):deploy.require_running_parent_handoff(self.work,changed)
    def test_python38_helpers(self):
        for name in ('advance_y102.py','y102_deployment.py','y101_y102_terminal_contract.py','prepare_y101_y102_terminal_binding.py','test_y102_admission.py'):
            ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'),feature_version=(3,8))
