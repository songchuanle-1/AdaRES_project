"""Reject malformed actual Y101 evidence, changed protocols and conflicting resources."""
import ast,copy,unittest
from prepare_y60_stage_heads import ROOT,monitor,ref,tick
from check_ef_execution import validate_frozen
import advance_y101 as advance
import y101_deployment as deploy

class Admission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=tick.load(ROOT/'pipeline/research_work.json');cls.c,cls.plan,_,cls.f=advance.prepared(cls.work)
        cls.raw=tick.load(ROOT/cls.c['native_smoke_result']['path'])
        cls.parentref=next(r for r in cls.work['candidates'] if r['id']=='AdaRES-Y99')['post_training_freeze'];cls.parent=tick.load(ROOT/cls.parentref['path'])
    def check(self,v=None):advance.validate_result(self.raw if v is None else v,self.c,self.f)
    def test_actual_native_and_exact_additions(self):
        self.check();self.assertEqual((self.f['runtime_server'],self.f['runtime_power_limit_watts']),('server-1',350.0));self.assertEqual(validate_frozen(self.c,self.plan,self.f),7)
        self.assertEqual(deploy.git('rev-parse',deploy.Y101+'^').decode().strip(),deploy.Y100)
        rows=[['A',p] for p in deploy.paths('AdaRES-Y101')];deploy.validate_delta(rows,'AdaRES-Y101')
        for bad in (rows[:-1],rows+[['A','foreign.py']],[['M',rows[0][1]]]+rows[1:]):
            with self.assertRaises(AssertionError):deploy.validate_delta(bad,'AdaRES-Y101')
        self.assertEqual(list(deploy.contract(deploy.Y100,deploy.Y101,'AdaRES-Y100')),[deploy.Y100,deploy.Y101])
    def test_exact_identity(self):
        for k,v in [('id','wrong'),('config','wrong.yaml'),('code_commit','0'*40),('goal_sha256','0'*64),('native_YAML_sha256','0'*64)]:
            with self.subTest(key=k),self.assertRaises(AssertionError):self.check(dict(self.raw,**{k:v}))
    def test_two_finite_real_train_steps(self):
        for k,v in [('split','val'),('native_entry','synthetic'),('steps',1),('optimizer_steps',1),('losses',[1.]),('losses',[1.,float('nan')]),('validation_used',True),('model_weights_saved',True),('admission',True)]:
            with self.subTest(key=k),self.assertRaises(AssertionError):self.check(dict(self.raw,**{k:v}))
    def test_required_paths_and_modes(self):
        for k in ['complete','native_data_loaded','backward_finite','intended_path_observed','static_mask_features_preserved','original_hard_gate_algorithm_preserved','native_modes_verified']:
            with self.subTest(key=k),self.assertRaises(AssertionError):self.check(dict(self.raw,**{k:False}))
        for k in ['query_FFN_skip_calls','original_query_FFN_calls','coarse_gate_consumers','shared_head_calls','query_norm_consumption_calls']:
            with self.subTest(key=k),self.assertRaises(AssertionError):self.check(dict(self.raw,**{k:0}))
    def test_original_gradients_and_new_updates(self):
        for k,v in [('all_finite',False),('full_input_base_adapted_and_original_FFN_gradients_finite',False),('original_shared_fusion_norm_gradients_finite',False),('static_mask_gradients_finite',False),('frozen_text_encoder_preserved',False),('original_query_FFN_parameter_tensors',0),('original_query_FFN_calls',0),('new_gradient_max',0.),('parameter_update_max',float('nan'))]:
            changed=copy.deepcopy(self.raw);changed['gradient_checks'][1][k]=v
            with self.subTest(key=k),self.assertRaises(AssertionError):self.check(changed)
    def test_full_current_query_and_original_complete_ffns(self):
        for k,v in [('original_full256_FFN_input_identity',False),('complete_original_FFN_output_identity',False),('original_shared_fusion_norm_consumed',False),('both_inputs_undetached',False),('float64_full_query_projection_reference',False),('prediction_consumes_adapted_query',False),('original_static_mask_preserved',False),('weight_nonzero',False),('query_delta_max',0.)]:
            changed=copy.deepcopy(self.raw);changed['query_FFN_paths'][6][k]=v
            with self.subTest(key=k),self.assertRaises(AssertionError):self.check(changed)
    def test_exact_train_data_distinct_scenes_and_memory(self):
        changed=copy.deepcopy(self.raw);changed['data_source']['config_data_root']='/wrong/data'
        with self.assertRaises(AssertionError):self.check(changed)
        changed=copy.deepcopy(self.raw);changed['batches'][1]=copy.deepcopy(changed['batches'][0])
        with self.assertRaises(AssertionError):self.check(changed)
        for v in (0,24*2**30,float('nan')):
            with self.assertRaises(AssertionError):self.check(dict(self.raw,peak_reserved_bytes=v))
    def test_complete_original_server1_protocol(self):
        before=copy.deepcopy(self.parent);frozen=advance.make_freeze(self.parent,self.parentref);self.assertEqual(before,self.parent)
        self.assertEqual(frozen['training'],before['training']);restored=copy.deepcopy(frozen['evaluation']);restored['methods'][0]='AdaRES-Y99';self.assertEqual(restored,before['evaluation']);self.assertEqual(frozen['resource_condition'],before['resource_condition']);self.assertEqual(frozen['resource_authorization'],ref(advance.GRANT))
        for k,v in [('id','wrong'),('code_commit','0'*40),('complete',False)]:
            with self.assertRaises(AssertionError):advance.make_freeze(dict(before,**{k:v}),self.parentref)
    def test_pause_private_runtime_busy_and_reserved(self):
        servers=monitor.load_servers(str(ROOT));self.assertEqual(servers['server-1']['allowed_gpus'],[0]);self.assertEqual(servers['server-2']['allowed_gpus'],[0]);self.assertEqual(servers['server-6']['allowed_gpus'],[]);self.assertEqual(ref(advance.PAUSE)['sha256'],advance.PAUSE_SHA)
        for k,v in [('python','/wrong/python'),('expected_hostname','foreign'),('allowed_gpus',[0,1]),('runtime_path_mappings',[])]:
            with self.assertRaises(AssertionError):advance.runtime('server-1',self.c,self.f,dict(servers['server-1'],**{k:v}))
        advance.idle(dict(gpu='0,1,350',compute=''),350.)
        for gpu,pid in [('nan,0,350',''),('0,0,280',''),('0,2000,350',''),('0,0,350','99')]:
            with self.assertRaises(AssertionError):advance.idle(dict(gpu=gpu,compute=pid),350.)
        queue,running=monitor.load_queue(str(ROOT)),monitor.load_running(str(ROOT));advance.unreserved(queue,running,'server-1',0)
        with self.assertRaises(AssertionError):advance.unreserved(queue,running,'server-2',0)
    def test_completed_owner_and_actual_native_handoff(self):
        advance.require_completed_owner();deploy.require_running_parent_handoff(self.work,self.c)
        changed=copy.deepcopy(self.c);changed['native_smoke_result']=dict(path='missing.json',sha256='0'*64)
        with self.assertRaises(AssertionError):deploy.require_running_parent_handoff(self.work,changed)
    def test_python38_helpers(self):
        for p in ['advance_y101.py','y101_deployment.py','y100_y101_terminal_contract.py','prepare_y100_y101_terminal_binding.py','test_y101_admission.py']:ast.parse((ROOT/'scripts'/p).read_text(encoding='utf-8'),feature_version=(3,8))
if __name__=='__main__':unittest.main()
