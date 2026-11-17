"""Reuse the frozen trained serial/stress cost protocol on Y101's original server1GPU0/350W."""
import argparse,ast,copy,hashlib,json,subprocess,sys
from datetime import datetime
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_frozen_review_job import remote
import closure

COST_SHA='e3859b80ef211cf7d0d0d15a1cfc4441288935a9098a2b71256602ebf189e035'

class IdentityAndMarkers(ast.NodeTransformer):
    def __init__(self):self.parameters=0;self.identity=0;self.markers={'serial':0,'stress':0}
    def visit_Constant(self,node):
        if node.value==164958087:self.parameters+=1;node.value=164695173
        elif node.value=='AdaRES-Y35':self.identity+=1;node.value='AdaRES-Y101'
        return node
    def visit_Expr(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='readings' and node.value.func.attr=='append' and node.value.args and isinstance(node.value.args[0],ast.Call) and isinstance(node.value.args[0].func,ast.Name) and node.value.args[0].func.id=='dict':
            names={k.arg for k in node.value.args[0].keywords}
            if 'scan_ids' in names and 'readings' in names:
                phase='serial' if 'dataset_index' in names else 'stress';self.markers[phase]+=1
                return [node,ast.parse("mark("+repr(phase)+",name,"+('index' if phase=='serial' else 'None')+")").body[0]]
        return node

HEADER=r'''import datetime,fcntl,hashlib,json,os,pathlib,subprocess,sys,time,traceback
folder=pathlib.Path(P['folder']);assert str(folder.resolve()).startswith('/data1/songcl/Project/AdaRES_project/pipeline/cost_jobs/AdaRES-Y101/') and folder.is_dir()
COUNT=0
def state(value):
    value.update(id='AdaRES-Y101',token=P['token'],pid=os.getpid(),native_cost_batches_done=COUNT,observed_at=datetime.datetime.now().astimezone().isoformat());temporary=folder/'state.next.json';temporary.write_text(json.dumps(value));os.replace(str(temporary),str(folder/'state.json'))
def mark(phase,method,index):
    global COUNT
    COUNT+=1;value=dict(status='measuring',phase=phase,method=method,dataset_index=index);state(value);print(json.dumps(dict(phase=phase,method=method,dataset_index=index,native_cost_batches_done=COUNT)),flush=True)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def source_check():
    root=pathlib.Path(P['root']);assert (root/'.codex_source_commit').read_text().strip()==P['code_commit'];rows=[]
    assert not pathlib.Path('/proc/2263581').exists()
    for row in P['all_source_files']:
        path=root/row['path'];raw=path.read_bytes();protected=raw.replace(b'\r\n',b'\n') if row['LF'] else raw
        assert hashlib.sha256(protected).hexdigest()==row['sha256'],row['path'];rows.append(dict(path=str(path),raw_sha256=hashlib.sha256(raw).hexdigest()))
    for row in P['runtime_files']:assert sha(row['path'])==row['sha256']
    return rows
try:
    state(dict(status='guarding'))
    assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73' and P['gpu']==0 and P['power']==350.0
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0' and sys.executable==P['python'] and sys.flags.no_site==1
    gpu_lease=open('/data1/songcl/Project/AdaRES_project/.foreground_gpu0_diagnostic.lock','a');fcntl.flock(gpu_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(P['root']);sys.path[:0]=P['python_paths']
    before_sources=source_check()
    captures={}
    for name,row in P['capture_files'].items():
        assert sha(row['path'])==row['sha256'];captures[name]=json.loads(pathlib.Path(row['path']).read_text());assert captures[name]['complete'] and captures[name]['completed_passes']==2
    assert hashlib.sha256(P['imports'].encode()).hexdigest()==P['imports_sha256'];exec(compile(P['imports'],'original_cost_imports','exec'),globals())
    assert not torch.cuda.is_initialized()
    power=float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip());assert power==350.
    assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
    used,util=[float(v.strip()) for v in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')];assert used<1024 and util<10
    root=pathlib.Path(P['root']);source_manifest=P['candidate_source_manifest']
    state(dict(status='loading_original_TRAIN_cost_data'))
    assert hashlib.sha256(P['core'].encode()).hexdigest()==P['core_sha256'];exec(compile(P['core'],'original_frozen_cost_science','exec'),globals())
    assert COUNT==24 and result['complete'] and result['id']=='AdaRES-Y101'
    assert result['parameters']==dict(candidate=164695173,parent=164629637)
    assert result['manifest_indices'][-2:]==[728,6780] and len(result['manifest_indices'])==10
    assert len(result['rows']['candidate'])==len(result['rows']['parent'])==10 and len(result['training_stress']['candidate'])==len(result['training_stress']['parent'])==2
    state(dict(status='postaudit',measured_methods=['candidate','parent']))
    after_sources=source_check();assert after_sources==before_sources
    checkpoint_audit=[]
    for row in P['checkpoint_audit']:
        path=pathlib.Path(row['path']);before=path.stat();actual=sha(path);after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and actual==row['sha256'];checkpoint_audit.append(dict(path=str(path),sha256=actual,bytes=before.st_size,unchanged_during_read=True))
    audit=dict(complete=True,id='AdaRES-Y101',token=P['token'],cost_file=P['output'],cost_sha256=sha(P['output']),native_cost_batches_done=COUNT,source_raw_bytes_preserved=True,source_files=after_sources,checkpoints=checkpoint_audit,original_training_absent=True,original_resource=dict(server='server-1',gpu=0,power_limit_watts=350.0),memory_only_stress_updates=True,training_restarted=False,original_checkpoints_overwritten=False,scientific_core_equivalent_after_identity_and_outside_timing_markers=True)
    (folder/'postaudit.json').write_text(json.dumps(audit));state(dict(status='complete',cost_file=P['output'],postaudit=str(folder/'postaudit.json')))
except BaseException as error:
    state(dict(status='failed',error=repr(error)));traceback.print_exc();raise
'''

def build(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');assert c['stage']=='validation' and not c.get('frozen_trained_cost_job')
    formal=tick.load(closure.artifact(ROOT,c['full_frozen_evaluation_result']));assert formal['complete'] and formal['full_passes_verified']==8
    job=c['frozen_review_job'];ep=closure.artifact(ROOT,job['program']);p=ast.literal_eval(ast.parse(ep.read_text(encoding='utf-8')).body[0].value)
    source=ROOT/'scripts/probe_y35_trained_cost_fixed.py';assert ref(source)['sha256']==COST_SHA
    tree=ast.parse(source.read_text(encoding='utf-8'));start=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cfg' for t in n.targets))
    original=ast.Module(body=tree.body[start:],type_ignores=[]);changed=copy.deepcopy(original);transform=IdentityAndMarkers();changed=transform.visit(changed);ast.fix_missing_locations(changed);assert transform.parameters==transform.identity==1 and transform.markers==dict(serial=1,stress=1)
    class Undo(ast.NodeTransformer):
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='mark':return None
            return self.generic_visit(n)
        def visit_Constant(self,n):
            if n.value==164695173:n.value=164958087
            elif n.value=='AdaRES-Y101':n.value='AdaRES-Y35'
            return n
    assert ast.dump(Undo().visit(copy.deepcopy(changed)),include_attributes=False)==ast.dump(original,include_attributes=False)
    core=ast.unparse(changed)+'\n';imports=ast.unparse(ast.Module(body=[n for n in tree.body[:start] if isinstance(n,(ast.Import,ast.ImportFrom))],type_ignores=[]))+'\n';ast.parse(core,feature_version=(3,8))
    folder='/data1/songcl/Project/AdaRES_project/pipeline/cost_jobs/AdaRES-Y101/'+token
    capturedir=ROOT/c['frozen_review_capture_directory'];captures={name:dict(path=job['folder']+'/'+cid.lower()+'.json',sha256=ref(capturedir/(cid.lower()+'.json'))['sha256']) for name,cid in (('candidate','AdaRES-Y101'),('parent','AdaRES-W2'))}
    terminal=tick.load(closure.artifact(ROOT,c['terminal_evidence']));staging=tick.load(closure.artifact(ROOT,c['frozen_review_reference_stage']));checkpoint_audit=[dict(path=r['path'],sha256=r['sha256']) for r in terminal['checkpoints']]+[dict(path=r['mapped_checkpoint'],sha256=r['expected_sha256']) for r in staging['references']]
    origin=tick.load(closure.artifact(ROOT,c['frozen_review_pointnet_origin']));wrapper=next(r for r in origin['wrappers'] if r['exists']);runtimefiles=[dict(path=wrapper['resolved'],sha256=wrapper['sha256'])]+[dict(path=r['path'],sha256=r['sha256']) for r in origin['original_live_PID_pointnet_binaries']]
    candidate_manifest=[(r['path'],r['sha256']) for r in p['source_files'] if not r['LF'] and 'adares_y101_query_ffn_skip/' in r['path']];assert len(candidate_manifest)==6
    candidate_manifest.append((p['candidate_config'],p['methods'][0]['config_sha256']))
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0]
    payload=dict(id='AdaRES-Y101',token=token,root=server['remote_code_root'],code_commit=p['code_commit'],gpu=0,power=350.,python=server['python'],python_paths=p['python_paths'],folder=folder,output=folder+'/trained_cost.json',capture_files=captures,all_source_files=p['source_files'],runtime_files=runtimefiles,candidate_source_manifest=candidate_manifest,checkpoint_audit=checkpoint_audit,imports=imports,imports_sha256=hashlib.sha256(imports.encode()).hexdigest(),core=core,core_sha256=hashlib.sha256(core.encode()).hexdigest())
    program='P='+repr(payload)+'\n'+HEADER;ast.parse(program,feature_version=(3,8));path=ROOT/('pipeline/checks/y101_frozen_cost_'+token+'.py');plan=ROOT/('pipeline/checks/y101_frozen_cost_'+token+'.json');assert not path.exists() and not plan.exists();path.write_text(program,encoding='utf-8',newline='\n')
    tick.save(plan,dict(complete=True,token=token,id='AdaRES-Y101',program=ref(path),original_cost_source=ref(source),scientific_AST_equivalent_after_candidate_identity_parameters_and_outside_timing_progress_markers=True,markers_outside_all_measurement_intervals=True,inference_protocol='Original first8 unique TRAIN scenes plus indices728/6780, batch1, augment/elastic disabled, seed1999, warm2/read3',native_stress_protocol='Original two predetermined TRAIN batch2 groups, fresh optimizer, warm2/read3;20 memory-only updates total; no checkpoint writes',resource=p['freeze']['resource_condition'],launch_started=False,checkpoint_loaded=False,GPU_used=False,evidence=[ref(__file__),ref(source),ref(path),c['full_frozen_evaluation_result'],c['post_training_freeze'],c['terminal_evidence'],c['frozen_review_runtime_binding']]))
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');assert not c.get('frozen_trained_cost_job');c['frozen_trained_cost_plan']=ref(plan);c['evidence'].extend([ref(plan),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    print(json.dumps(dict(plan=ref(plan),scientific_AST_preserved=True,GPU_used=False)))

def launch(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');assert c['stage']=='validation' and not c.get('frozen_trained_cost_job') and not c.get('frozen_trained_cost_launch_intent')
    plan=tick.load(closure.artifact(ROOT,c['frozen_trained_cost_plan']));assert plan['complete'];program_path=closure.artifact(ROOT,plan['program']);text=program_path.read_text(encoding='utf-8');p=ast.literal_eval(ast.parse(text).body[0].value)
    q=monitor.load_queue(str(ROOT));assert not monitor.load_running(str(ROOT)).get('server-1',{}).get('0') and not any(r['status'] in ('running','claiming') and str(r.get('gpu'))=='0' for r in q['gpu_queues']['server-1'])
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0]
    intent=ROOT/('pipeline/checks/y101_cost_launch_intent_'+token+'.json');assert not intent.exists();tick.save(intent,dict(complete=True,token=token,id='AdaRES-Y101',program=ref(program_path),plan=c['frozen_trained_cost_plan'],remote_folder=p['folder'],status='intent_before_remote_creation',original_review_folder=c['frozen_review_job']['folder']))
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');c['frozen_trained_cost_launch_intent']=ref(intent);c['evidence'].append(ref(intent));tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    body=r'''import hashlib,json,os,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
assert not pathlib.Path('/proc/2263581').exists();assert json.loads((pathlib.Path(Q['review_folder'])/'state.json').read_text())['status']=='complete'
assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
assert float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip())==350.
folder=pathlib.Path(Q['folder']);assert str(folder.resolve()).startswith('/data1/songcl/Project/AdaRES_project/pipeline/cost_jobs/AdaRES-Y101/') and not folder.exists();folder.mkdir(parents=True)
raw=Q['program'].encode();assert hashlib.sha256(raw).hexdigest()==Q['sha'];path=folder/'worker.py';path.write_bytes(raw)
(folder/'launch_intent.json').write_text(json.dumps(dict(token=Q['token'],status='launching',program_sha256=Q['sha'])))
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=Q['library'],PYTHONPATH=os.pathsep.join(Q['paths']))
with (folder/'worker.log').open('xb') as log:child=subprocess.Popen([Q['python'],'-B','-S',str(path)],cwd=Q['root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
proof=dict(complete=True,id='AdaRES-Y101',token=Q['token'],pid=child.pid,folder=str(folder),program_sha256=Q['sha'],process_created=True,native_progress_confirmed=False)
(folder/'launch.json').write_text(json.dumps(proof));print(json.dumps(proof))
'''
    out=ROOT/('pipeline/checks/y101_cost_launch_'+token+'.json');value=remote(server,'Q='+repr(dict(folder=p['folder'],review_folder=c['frozen_review_job']['folder'],token=token,program=text,sha=ref(program_path)['sha256'],python=server['python'],library=server['runtime_library_path'],paths=p['python_paths'],root=server['remote_code_root']))+'\n'+body,out)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');c['frozen_trained_cost_job']=dict(folder=p['folder'],token=token,pid=value['pid'],launch=ref(out),program=ref(program_path),native_progress_confirmed=False);c['evidence'].extend([ref(out),ref(out.with_suffix('.log')),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    print(json.dumps(dict(launch=ref(out),pid=value['pid'],native_progress_confirmed=False)))

def observe(token,label):
    lease(token,300);w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');job=c['frozen_trained_cost_job'];server=monitor.load_servers(str(ROOT))['server-1']
    body=r'''import json,pathlib,subprocess,time
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
folder=pathlib.Path(Q['folder'])
def read():
    v=json.loads((folder/'state.json').read_text());assert v['pid']==Q['pid'] and v['token']==Q['token'];return v
a=read();time.sleep(5);b=read();compute=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
with (folder/'worker.log').open('rb') as f:
 f.seek(max(0,(folder/'worker.log').stat().st_size-6000));tail=f.read().decode('utf-8',errors='replace')
print(json.dumps(dict(complete=True,id='AdaRES-Y101',before=a,after=b,compute_pids=[int(n) for n in compute.splitlines() if n.strip()],native_progress_confirmed=b['native_cost_batches_done']>a['native_cost_batches_done'] or b['status']=='complete',log_tail=tail,original_training_absent=not pathlib.Path('/proc/2263581').exists(),remote_mutations=False)))
'''
    out=ROOT/('pipeline/checks/y101_cost_progress_'+token+'_'+label+'.json');value=remote(server,'Q='+repr(job)+'\n'+body,out)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');c['frozen_trained_cost_job']['latest_observation']=ref(out);c['frozen_trained_cost_job']['native_progress_confirmed']=value['native_progress_confirmed'] or c['frozen_trained_cost_job'].get('native_progress_confirmed',False);c['evidence'].extend([ref(out),ref(out.with_suffix('.log')),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    print(json.dumps(dict(observation=ref(out),state=value['after'],native_progress_confirmed=value['native_progress_confirmed'],compute_pids=value['compute_pids'],log_tail=value['log_tail'][-2500:])))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('build','launch','observe'));parser.add_argument('--token',required=True);parser.add_argument('--label',default='start');a=parser.parse_args()
    if a.operation=='build':build(a.token)
    elif a.operation=='launch':launch(a.token)
    else:observe(a.token,a.label)
