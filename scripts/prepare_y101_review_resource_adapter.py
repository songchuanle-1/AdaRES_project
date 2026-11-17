"""Prepare a server1 GPU0-only resource adapter; preserve the entire evaluator science."""
import argparse,ast,json
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from reconcile_y62_y63_server1 import remote
from y102_deployment import git,Y102
import closure

ORIGINAL_SHA='7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13'
OLD_LOCK='/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock'
NEW_LOCK='/data1/songcl/Project/AdaRES_project/.foreground_gpu0_diagnostic.lock'
OLD_CLI='parser.add_argument("--gpu-id", type=int, default=1)'
NEW_CLI='parser.add_argument("--gpu-id", type=int, default=0, choices=(0,))'
HOST='''    resource_hostname = subprocess.check_output(["hostname"], universal_newlines=True).strip()
    if resource_hostname != "dl02-MS-7B73":
        raise RuntimeError("WRONG_ORIGINAL_SERVER")
'''
RESOURCE='''    resource_power = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=power.limit", "--format=csv,noheader,nounits"], universal_newlines=True).strip()
    if float(resource_power) != 350.0:
        raise RuntimeError("WRONG_ORIGINAL_POWER")
    resource_compute = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], universal_newlines=True).strip()
    if resource_compute:
        raise RuntimeError("GPU_COMPUTE_OCCUPIED: " + resource_compute)
'''
TESTS=r'''import argparse,ast,contextlib,fcntl,io,json,pathlib,subprocess,sys,tempfile,types,unittest
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
original=P['original'];adapted=P['adapted']
module=ast.parse(adapted);main=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='main')
start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='resource_hostname' for t in n.targets))
end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='set_num_threads')
selected=main.body[start:end]
function=ast.FunctionDef(name='guard',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='args')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=selected+[ast.Return(value=ast.Tuple(elts=[ast.Name(id='lease',ctx=ast.Load()),ast.Name(id='used',ctx=ast.Load()),ast.Name(id='util',ctx=ast.Load())],ctx=ast.Load()))],decorator_list=[])
compiled=compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'actual_adapter_resource_prefix','exec')
parser_body=[]
for statement in main.body:
    parser_body.append(statement)
    if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in statement.targets):break
parser_code=compile(ast.fix_missing_locations(ast.Module(body=parser_body,type_ignores=[])),'actual_adapter_parser','exec')
class Guards(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='y101_adapter_'+P['token']+'_',dir=P['test_directory']);self.handles=[];self.queries=[]
        self.host='dl02-MS-7B73\n';self.power='350.00\n';self.compute='';self.occupation='0, 0\n';self.failure=False
    def tearDown(self):
        for handle in self.handles:handle.close()
        self.temp.cleanup()
    def fake(self,argv,**kwargs):
        self.queries.append(argv)
        if argv==['hostname']:return self.host
        assert argv[:3]==['nvidia-smi','-i','0']
        if self.failure:raise subprocess.CalledProcessError(1,argv)
        query=argv[3]
        if query=='--query-gpu=power.limit':return self.power
        if query=='--query-compute-apps=pid':return self.compute
        assert query=='--query-gpu=memory.used,utilization.gpu'
        return self.occupation
    def opened(self,path,mode):
        assert path==P['new_lock'] and mode=='a'
        handle=open(str(pathlib.Path(self.temp.name)/'test_only.lock'),mode);self.handles.append(handle);return handle
    def guard(self):
        ns=dict(subprocess=types.SimpleNamespace(check_output=self.fake),fcntl=fcntl,open=self.opened)
        exec(compiled,ns);return ns['guard'](types.SimpleNamespace(gpu_id=0))
    def parse(self,gpu=None):
        old=sys.argv;sys.argv=['test','--config','unchanged.yaml','--checkpoint','unloaded.pth','--output','unwritten.json']+([] if gpu is None else ['--gpu-id',str(gpu)])
        try:
            ns={'argparse':argparse}
            with contextlib.redirect_stderr(io.StringIO()):exec(parser_code,ns)
            return ns['args']
        finally:sys.argv=old
    def test_scientific_AST_preserved(self):
        restored=adapted.replace(P['host'],'').replace(P['resource'],'').replace(P['new_lock'],P['old_lock']).replace(P['new_cli'],P['old_cli'])
        self.assertEqual(ast.dump(ast.parse(restored),include_attributes=False),ast.dump(ast.parse(original),include_attributes=False))
    def test_default_gpu0(self):self.assertEqual(self.parse().gpu_id,0)
    def test_explicit_gpu0(self):self.assertEqual(self.parse(0).gpu_id,0)
    def test_gpu1_rejected_before_imports(self):
        with self.assertRaises(SystemExit):self.parse(1)
        self.assertEqual(self.queries,[]);self.assertEqual(self.handles,[])
    def test_other_gpu_rejected(self):
        with self.assertRaises(SystemExit):self.parse(6)
    def test_original_resource_accepted(self):
        handle,used,util=self.guard();self.assertEqual((used,util),(0,0));self.assertFalse(handle.closed)
    def test_wrong_host_rejected_before_lock(self):
        self.host='wrong-host\n'
        with self.assertRaisesRegex(RuntimeError,'WRONG_ORIGINAL_SERVER'):self.guard()
        self.assertEqual(self.handles,[])
    def test_wrong_power_rejected(self):
        self.power='280\n'
        with self.assertRaisesRegex(RuntimeError,'WRONG_ORIGINAL_POWER'):self.guard()
    def test_live_compute_rejected_even_zero_util(self):
        self.compute='2263581\n'
        with self.assertRaisesRegex(RuntimeError,'GPU_COMPUTE_OCCUPIED'):self.guard()
    def test_busy_memory_rejected(self):
        self.occupation='1024, 0\n'
        with self.assertRaisesRegex(RuntimeError,'GPU_NOT_IDLE'):self.guard()
    def test_busy_utilization_rejected(self):
        self.occupation='0, 10\n'
        with self.assertRaisesRegex(RuntimeError,'GPU_NOT_IDLE'):self.guard()
    def test_failed_query_rejected(self):
        self.failure=True
        with self.assertRaises(subprocess.CalledProcessError):self.guard()
    def test_contending_lock_rejected(self):
        self.guard()
        with self.assertRaises(BlockingIOError):self.guard()
    def test_missing_power_rejected(self):
        self.power=''
        with self.assertRaises(ValueError):self.guard()
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
assert 'torch' not in sys.modules and 'gorilla' not in sys.modules
print(json.dumps(dict(complete=True,passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),test_log=stream.getvalue(),real_GPU_queries=False,real_GPU_lock_acquired=False,checkpoint_loaded=False,CUDA_used=False,scientific_evaluation_executed=False,temporary_directory_cleaned=True)))
assert result.wasSuccessful()
'''

def prepare(token):
    lease(token,600);assert git('rev-parse','HEAD').decode().strip()==Y102
    immutable={p:ref(p) for p in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/admissions/AdaRES-Y101.json','pipeline/admissions/AdaRES-Y102.json')};index=git('ls-files','--stage','-z')
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['queue']['candidate_budget']==view['research']['creation_budget']==view['research']['reserve_budget']==0
    work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert candidate['stage']=='training' and not candidate.get('frozen_review_resource_adapter')
    freeze=tick.load(closure.artifact(ROOT,candidate['post_training_freeze']));assert freeze['resource_condition']['server']=='server-1' and freeze['resource_condition']['gpu']==0 and freeze['resource_condition']['power_limit_watts']==350.0
    staging=tick.load(closure.artifact(ROOT,candidate['frozen_review_reference_stage']));assert staging['all_three_reference_bytes_match_frozen_SHA'] and not staging['evaluation_ready']
    source=ROOT/'AdaRES/tools/evaluate_adares_frozen_full.py';assert ref(source)['sha256']==ORIGINAL_SHA
    original=source.read_text(encoding='utf-8');assert original.count(OLD_CLI)==original.count(OLD_LOCK)==1
    lockline='    lease = open("'+OLD_LOCK+'", "a")\n';flock='    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
    assert original.count(lockline)==original.count(flock)==1
    adapted=original.replace(lockline,HOST+lockline).replace(flock,flock+RESOURCE).replace(OLD_LOCK,NEW_LOCK).replace(OLD_CLI,NEW_CLI)
    assert adapted.count(HOST)==adapted.count(RESOURCE)==1
    restored=adapted.replace(HOST,'').replace(RESOURCE,'').replace(NEW_LOCK,OLD_LOCK).replace(NEW_CLI,OLD_CLI)
    assert ast.dump(ast.parse(restored),include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False)
    ast.parse(adapted,feature_version=(3,8))
    base='pipeline/checks/y101_review_adapter_'+token;adapter=ROOT/(base+'.py');program=ROOT/(base+'_tests.py');result_path=ROOT/(base+'_tests.json');manifest=ROOT/(base+'.json')
    assert not any(p.exists() for p in (adapter,program,result_path,manifest))
    adapter.write_text(adapted,encoding='utf-8',newline='\n')
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0]
    payload=dict(original=original,adapted=adapted,host=HOST,resource=RESOURCE,old_cli=OLD_CLI,new_cli=NEW_CLI,old_lock=OLD_LOCK,new_lock=NEW_LOCK,token=token,test_directory='/data1/songcl/Project/AdaRES_project/pipeline')
    text='P='+repr(payload)+'\n'+TESTS;ast.parse(text,feature_version=(3,8));program.write_text(text,encoding='utf-8',newline='\n')
    value=remote(server,text,result_path);tick.save(result_path,value)
    assert value['passed'] and value['tests_run']==14 and not value['CUDA_used'] and not value['real_GPU_queries']
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert all(ref(p)==r for p,r in immutable.items()) and git('ls-files','--stage','-z')==index
        tick.save(manifest,dict(complete=True,token=token,id='AdaRES-Y101',prepared_at=datetime.now().astimezone().isoformat(),original_evaluator=ref(source),adapted_evaluator=ref(adapter),resource_condition=freeze['resource_condition'],tests=ref(result_path),tests_passed=14,scientific_AST_identical_after_reversing_only_resource_edits=True,resource_edits=['GPU0 default and choices0; reject GPU1 before imports','exact server1 hostname','original350W power check','shared existing GPU0 diagnostic lock','reject every compute PID including original live training'],original_idle_thresholds_unchanged=True,checkpoint_loaded=False,CUDA_used=False,real_GPU_queries=False,actual_resource_lock_acquired=False,adapter_deployed=False,evaluation_executed=False,evaluation_ready=False,remaining='Await natural original Y101 fixed30, exact full finite checkpoint proof/canonical archive, fresh exclusive authorized server1GPU0/350W, tested private-runtime/code import launch binding, full paired FIRST/bootstrap/subgroup/cost. No migration of any original GPU1 review.',evidence=[ref(__file__),ref(source),ref(adapter),ref(program),ref(result_path),ref(result_path.with_suffix('.log')),candidate['post_training_freeze'],candidate['frozen_review_reference_stage']]))
        work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert candidate['stage']=='training';candidate['frozen_review_resource_adapter']=ref(manifest);candidate['evidence'].extend([ref(manifest),ref(__file__)])
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(adapter=ref(manifest),tests_passed=14,scientific_evaluation_executed=False,checkpoint_loaded=False,GPU_used=False,adapter_deployed=False)))
    finally:writer.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);prepare(p.parse_args().token)
