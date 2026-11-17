"""Admission rejection controls; only the recorded real smoke supplies evidence."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT,tick
import advance_y74 as advance
import y74_deployment as deploy

class Admission(unittest.TestCase):
    def setUp(self):
        work=tick.load(ROOT/'pipeline/research_work.json')
        self.c=copy.deepcopy(next(c for c in work['candidates'] if c['id']=='AdaRES-Y74'))
        plan=tick.load(ROOT/self.c['native_smoke_plan_server2_gpu0']['path'])
        self.f=ast.literal_eval(ast.parse((ROOT/plan['program']['path']).read_text(encoding='utf-8')).body[0].value)
        self.raw=tick.load(ROOT/self.c['native_smoke_result']['path'])
    def check(self,raw=None):advance.validate_result(self.raw if raw is None else raw,self.c,self.f)
    def test_real_recorded_native_result(self):self.check()
    def test_wrong_goal_version_config_or_id(self):
        for k,v in [('id','AdaRES-Y73'),('code_commit','0'*40),('config','other.yaml'),('goal_sha256','0'*64),('native_YAML_sha256','0'*64)]:
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(dict(self.raw,**{k:v}))
    def test_nonfinite_or_missing_losses(self):
        for v in ([float('nan'),1.],[1.,float('inf')],[1.]):
            with self.assertRaises(AssertionError):self.check(dict(self.raw,losses=v))
    def test_synthetic_validation_or_missing_native_path(self):
        for k,v in [('split','val'),('native_data_loaded',False),('backward_finite',False),('intended_path_observed',False),('native_modes_verified',False),('validation_used',True),('model_weights_saved',True)]:
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(dict(self.raw,**{k:v}))
    def test_missing_steps_heads_or_consumption(self):
        for k in ('steps','optimizer_steps','shared_head_calls','gate_calls','qqa_consumption_calls'):
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(dict(self.raw,**{k:0}))
    def test_bypassed_static_mask_or_current_visual(self):
        for k in ('native_qla_output_identity_verified','incoming_visual_query_identity_verified','both_inputs_undetached_verified','qqa_consumption_verified','static_mask_features_preserved','padding_identity_verified'):
            raw=copy.deepcopy(self.raw);raw['gate_paths'][7][k]=False
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(raw)
    def test_no_new_gradient_or_update(self):
        for k,v in [('gate_weight_gradient_max',0.),('parameter_update_max',0.),('language_and_incoming_query_gradients_finite',False)]:
            raw=copy.deepcopy(self.raw);raw['gradient_checks'][1][k]=v
            with self.subTest(k=k),self.assertRaises(AssertionError):self.check(raw)
    def test_wrong_original_data_or_duplicate_scenes(self):
        raw=copy.deepcopy(self.raw);raw['data_source']['config_data_root']='/wrong/data'
        with self.assertRaises(AssertionError):self.check(raw)
        raw=copy.deepcopy(self.raw);raw['batches'][1]=copy.deepcopy(raw['batches'][0])
        with self.assertRaises(AssertionError):self.check(raw)
    def test_resource_unknown_foreign_or_changed_power(self):
        for gpu,compute in [('nan,0,280',''),('0,0,350',''),('0,2000,280',''),('0,0,280','99')]:
            with self.assertRaises(AssertionError):advance.idle(dict(gpu=gpu,compute=compute),280.)
    def test_canonical_reservation_or_queued_work_refused(self):
        pause=dict(id='AdaRES-Y57',status='paused',gpu=1,dispatch_hold=True,resume_requires_user_authorization=True)
        queue=dict(gpu_queues={'server-2':[pause],'server-1':[]})
        advance.unreserved(queue,{},'server-1',0)
        for status in ('waiting','claiming','running'):
            q=copy.deepcopy(queue);q['gpu_queues']['server-1'].append(dict(id='foreign',status=status,gpu=0))
            with self.assertRaises(AssertionError):advance.unreserved(q,{},'server-1',0)
        with self.assertRaises(AssertionError):advance.unreserved(queue,{'server-1':{'0':'foreign'}},'server-1',0)
    def test_paused_y57_required(self):
        pause=dict(id='AdaRES-Y57',status='running',dispatch_hold=True,resume_requires_user_authorization=True)
        with self.assertRaises(AssertionError):advance.unreserved(dict(gpu_queues={'server-2':[pause]}),{},'server-1',0)
    def test_registered_chain_and_keeper_bytes(self):
        for cid,commit in (('AdaRES-Y72',deploy.Y72),('AdaRES-Y73',deploy.Y73)):
            result=deploy.contract(commit,deploy.Y74,cid)
            self.assertIn(deploy.Y74,result)
    def test_unknown_or_reverse_deployment_refused(self):
        for commit,head in ((deploy.Y72,'0'*40),('0'*40,deploy.Y74),(deploy.Y74,deploy.Y73)):
            with self.assertRaises(AssertionError):deploy.contract(commit,head,'AdaRES-Y72')
    def test_only_exact_additive_candidate_files(self):
        for cid in ('AdaRES-Y74',):
            rows=[['A',p] for p in deploy.paths(cid)];deploy.validate_delta(rows,cid)
            for bad in (rows+[['A','other.py']],[['M',rows[0][1]]]+rows[1:]):
                with self.assertRaises(AssertionError):deploy.validate_delta(bad,cid)

if __name__=='__main__':unittest.main()
