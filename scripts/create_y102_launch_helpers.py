"""Specialize actual-native Y102 admission/deployment and protect original Y101."""
import argparse,ast,json,re
from prepare_y60_stage_heads import ROOT,lease,ref,tick
from y101_deployment import Y100,Y101,git
from run_y102_bound_native_y101 import validate_binding,runtime
from y102_contract import validate_result
from check_ef_execution import validate_frozen
import closure
Y102='0c6c844719c8466ddd8888d054fe3b7ee658e643'

class Rewrite(ast.NodeTransformer):
    def __init__(self,changes,ints=None):
        self.changes=changes;self.ints=ints or {};self.pattern=re.compile('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)))
    def text(self,value):return self.pattern.sub(lambda m:self.changes[m.group()],value)
    def visit_Constant(self,n):
        if isinstance(n.value,str):n.value=self.text(n.value)
        elif type(n.value) is int:n.value=self.ints.get(n.value,n.value)
        return n
    def visit_Name(self,n):n.id=self.text(n.id);return n
    def visit_Attribute(self,n):n.attr=self.text(n.attr);return self.generic_visit(n)
    def visit_keyword(self,n):
        if n.arg:n.arg=self.text(n.arg)
        return self.generic_visit(n)
    def visit_ImportFrom(self,n):n.module=self.text(n.module);return self.generic_visit(n)
    def visit_alias(self,n):
        n.name=self.text(n.name)
        if n.asname:n.asname=self.text(n.asname)
        return n

def transformed(path,changes,functions=None,ints=None):
    tree=ast.parse((ROOT/path).read_text(encoding='utf-8'))
    if functions is not None:
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions];assert len(tree.body)==len(functions)
    return ast.unparse(ast.fix_missing_locations(Rewrite(changes,ints).visit(tree)))+'\n'

ADMISSION_TESTS='''"""Exercise operational guards against the actual Y102 native observations."""
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
'''

def create(token):
    lease(token,600);assert git('rev-parse','HEAD').decode().strip()==Y101 and git('rev-parse',Y102+'^').decode().strip()==Y101
    before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/research_work.json','pipeline/goal.json','SERVERS.yaml','pipeline/admissions/AdaRES-Y101.json','pipeline/admissions/AdaRES-Y100.json')};index=git('ls-files','--stage','-z')
    work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y102');assert c['stage']=='validation' and c['code_commit']==Y102 and not c.get('admission_receipt')
    validate_binding(tick.load(closure.artifact(ROOT,c['additive_version_binding'])),c,Y101)
    plan=tick.load(closure.artifact(ROOT,c['native_smoke_plan']));f=ast.literal_eval(ast.parse(closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(c,plan,f)==7;runtime('server-2',c,f,monitor_servers());validate_result(tick.load(closure.artifact(ROOT,c['native_smoke_result'])),c,f)
    changes={'run_y100_native_v2':'run_y102_bound_native_y101','advance_y100':'advance_y102','y100_deployment':'y102_deployment','y100_admission_contract':'y102_contract','test_y100_admission.py':'test_y102_admission.py','bcd19eedca805c8491568b3f81cf9cefa8dc4b0def8f6a09dd87c58f5e29d51c':'cb9b9a80fb6f9f9b116f9e9a35a515b3c4fb542eb6f4128cbee0ed5d389eff01','Y98':'Y100','y98':'y100','Y99':'Y101','y99':'y101','Y100':'Y102','y100':'y102'}
    advance=transformed('scripts/advance_y100.py',changes).replace('independent full1024 appearance bypass beside the original visual-memory projection MLP','independent full-scene point-mass-centered 3D geometry bypass beside the original complete static-mask path')
    header='''"""Select actual-native Y102 additions; preserve original Y101 training."""
import argparse,ast,hashlib,inspect,json,subprocess
from datetime import datetime
from pathlib import Path
import yaml
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from observe_y69_server1_phase import parse_phase,phase_key,progressed,REMOTE
import closure
import y101_deployment as previous
Y100,Y101=previous.Y100,previous.Y101
Y102="0c6c844719c8466ddd8888d054fe3b7ee658e643"
VERSIONS=previous.VERSIONS+(Y102,)
ADDITIONS=dict(previous.ADDITIONS,**{Y102:"AdaRES-Y102"})
git,adapt=previous.git,previous.adapt
def paths(cid):
    if cid!="AdaRES-Y102":return previous.paths(cid)
    from prepare_y102_geometry_mask_skip import files
    return [p.relative_to(ROOT).as_posix() for p in files()]
def require_running_parent_handoff(work,candidate):
    original=next(r for r in work["candidates"] if r["id"]=="AdaRES-Y101")
    assert original["stage"]=="training"
    plan=tick.load(closure.artifact(ROOT,original["terminal_handoff_plan"]))
    assert plan["complete"] is True and plan["id"]==original["id"]
    assert plan["tested_additive_native_evidence"]==candidate["native_smoke_result"]
    assert plan["payload"]["allowed_markers"]==[Y101,Y102]
    assert plan["CPU_guard_tests_passed"]==30 and plan["additive_guard_tests_passed"]==8
    for entry in plan["dependencies"]:closure.artifact(ROOT,entry)
'''
    pair={'advance_y101':'advance_y102','Y100':'Y101','Y101':'Y102','y100':'y101','y101':'y102'}
    deploy=header+transformed('scripts/y101_deployment.py',pair,{'validate_delta','select','observe'})+transformed('scripts/y101_deployment.py',{'UNUSED':'UNUSED'},{'contract'})
    deploy+='''
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("operation",choices=("select","observe"));p.add_argument("--id",choices=("AdaRES-Y101","AdaRES-Y102"),default="AdaRES-Y102");p.add_argument("--token",required=True);p.add_argument("--label",default="progress");a=p.parse_args()
    lease(a.token,240);writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:select(a.token) if a.operation=="select" else observe(a.token,a.id,a.label)
    finally:writer.close()
'''
    binding_changes={'y99_terminal_contract':'y101_terminal_contract','y100_deployment':'y102_deployment','advance_y100':'advance_y102','y100_admission_contract':'y102_contract','Y98':'Y100','y98':'y100','Y99':'Y101','y99':'y101','Y100':'Y102','y100':'y102'}
    binding=transformed('scripts/y99_y100_terminal_contract.py',binding_changes)
    test_changes=dict(binding_changes,**{'y99_y100_terminal_contract':'y101_y102_terminal_contract','20261111_180638':'20261112_200238','4d87ad8b3a6efe78be8dc9e8214bba9226160a22b649f64f8e1dd26d73988ede':'379a5b050624141f20345f614877f062f1a61309b55531fe977e023538fecd0c','fd07c0536891d905a7cd89e2b4211b32ed31a532b7b70ac9d71a8d7a66e97dbc':'e624a80505b35aa2fa09fa55bde64ed8f8e1915dbe385da3073db5b44bc81db1'})
    tests=transformed('scripts/test_y99_y100_terminal_binding.py',test_changes,ints={2980836:2263581})
    prep=transformed('scripts/prepare_y100_y101_terminal_binding.py',{'y100':'y101','y101':'y102','Y100':'Y101','Y101':'Y102'},ints={29:30})
    prep=prep.replace('pipeline/checks/y101_server2_fixed30_','pipeline/checks/y101_server1_fixed30_')
    outputs={'scripts/advance_y102.py':advance,'scripts/y102_deployment.py':deploy,'scripts/y101_y102_terminal_contract.py':binding,'scripts/test_y101_y102_terminal_binding.py':tests,'scripts/prepare_y101_y102_terminal_binding.py':prep,'scripts/test_y102_admission.py':ADMISSION_TESTS}
    sources=['scripts/advance_y100.py','scripts/y101_deployment.py','scripts/y99_y100_terminal_contract.py','scripts/test_y99_y100_terminal_binding.py','scripts/prepare_y100_y101_terminal_binding.py'];originals=[ref(p) for p in sources]
    for path,body in outputs.items():assert not (ROOT/path).exists(),path;ast.parse(body,feature_version=(3,8))
    for path,body in outputs.items():(ROOT/path).write_text(body,encoding='utf-8',newline='\n')
    assert [ref(p) for p in sources]==originals and all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
    out=ROOT/('pipeline/checks/y102_launch_helpers_'+token+'.json');assert not out.exists();tick.save(out,dict(complete=True,token=token,actual_native=c['native_smoke_result'],generated=[ref(p) for p in outputs],original_inputs=originals,scientific_files_and_main_index_unchanged=True,guard_tests_pending=True))
    print(json.dumps(dict(manifest=ref(out),generated=len(outputs),guards_pending=True)))

def monitor_servers():
    from prepare_y60_stage_heads import monitor
    return monitor.load_servers(str(ROOT))['server-2']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);create(p.parse_args().token)
