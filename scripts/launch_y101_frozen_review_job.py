"""Launch/observe only the archived original Y101 full review on its original GPU0."""
import argparse,ast,hashlib,json,shlex,subprocess,sys
from datetime import datetime
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y102_deployment import Y101,Y102,git
from y101_frozen_review_job import WORKER,STRIP_BOOTSTRAP,remote
import closure

def test(token):
    lease(token,300);work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101')
    adapter=tick.load(closure.artifact(ROOT,c['frozen_review_resource_adapter']));source=closure.artifact(ROOT,adapter['adapted_evaluator']).read_text(encoding='utf-8')
    bootstrap="import ast,sys;P0="+repr(dict(paths=[],source=source))+";sys.path[:0]=P0['paths'];"+STRIP_BOOTSTRAP
    output=ROOT/('pipeline/checks/y101_review_bootstrap_'+token+'_must_not_exist.json');assert not output.exists()
    log=ROOT/('pipeline/checks/y101_review_bootstrap_'+token+'.log');assert not log.exists()
    r=subprocess.run([sys.executable,'-B','-S','-c',bootstrap,'--config','unread.yaml','--checkpoint','unloaded.pth','--output',str(output),'--gpu-id','1'],capture_output=True,timeout=30);log.write_bytes(r.stdout+r.stderr)
    assert r.returncode==2 and b'invalid choice' in r.stderr and not output.exists()
    proof=ROOT/('pipeline/checks/y101_review_bootstrap_'+token+'.json');assert not proof.exists();tick.save(proof,dict(complete=True,token=token,id='AdaRES-Y101',actual_child_bootstrap_GPU1_rejected_before_model_imports=True,returncode=2,checkpoint_loaded=False,GPU_used=False,evidence=[ref(__file__),ref('scripts/y101_frozen_review_job.py'),ref(log),c['frozen_review_resource_adapter']]))
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');c['frozen_review_bootstrap_test']=ref(proof);c['evidence'].extend([ref(proof),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    print(json.dumps(dict(bootstrap=ref(proof),GPU_used=False)))

def launch(token):
    lease(token,600);assert git('rev-parse','HEAD').decode().strip()==Y102
    work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert c['stage']=='validation' and c['scientific_status']=='fixed30_complete_frozen_review_pending' and not c.get('frozen_review_job')
    terminal=tick.load(closure.artifact(ROOT,c['terminal_evidence']));archive=tick.load(closure.artifact(ROOT,c['terminal_archive_binding']));assert archive['complete'] and archive['terminal_evidence']==c['terminal_evidence']
    assert monitor.load_experiment(str(ROOT),'AdaRES-Y101')['status']=='done_complete'
    queue=monitor.load_queue(str(ROOT));running=monitor.load_running(str(ROOT));assert not running.get('server-1',{}).get('0')
    assert not any(str(r.get('gpu'))=='0' and r['status'] in ('running','claiming') for r in queue['gpu_queues']['server-1'])
    assert not any(r['id']=='AdaRES-Y101' for r in queue['gpu_queues']['server-1'])
    paused=next(r for r in queue['gpu_queues']['server-2'] if r['id']=='AdaRES-Y57');assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    immutable={p:ref(p) for p in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/admissions/AdaRES-Y101.json','pipeline/admissions/AdaRES-Y102.json')};index=git('ls-files','--stage','-z')
    freeze=tick.load(closure.artifact(ROOT,c['post_training_freeze']));adapter=tick.load(closure.artifact(ROOT,c['frozen_review_resource_adapter']));runtime=tick.load(closure.artifact(ROOT,c['frozen_review_runtime_binding']));staging=tick.load(closure.artifact(ROOT,c['frozen_review_reference_stage']));plan=tick.load(closure.artifact(ROOT,c['frozen_review_worker_plan']));boot=tick.load(closure.artifact(ROOT,c['frozen_review_bootstrap_test']))
    assert plan['CPU_launch_guards_passed']==11 and boot['actual_child_bootstrap_GPU1_rejected_before_model_imports'] and adapter['tests_passed']==14 and runtime['isolated_CPU_imports_passed']==4
    assert closure.artifact(ROOT,plan['worker']).read_text(encoding='utf-8')==WORKER
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0]
    original=ast.literal_eval(ast.parse((ROOT/'pipeline/analyses/terminal_bindings_6debf88fed634a13b772f27bdc244bbd_program.py').read_text(encoding='utf-8')).body[0].value)
    packages={PurePosixPath(r['config']).stem.removesuffix('_res') for r in staging['references']}
    sources=[dict(path=r['path'],sha256=r['lf_sha256'],LF=True) for r in original['sources'] if PurePosixPath(r['path']).parts[2] in packages];assert len(sources)==18
    folder=ROOT/'AdaRES/ipdn/model_paper5/adares_y101_query_ffn_skip';sources.extend(dict(path=p.relative_to(ROOT/'AdaRES').as_posix(),sha256=ref(p)['sha256'],LF=False) for p in folder.glob('*.py'));assert len(sources)==24
    for path in ('tools/train.py','tools/evaluate_adares_frozen_full.py'):
        raw=git('show',Y101+':AdaRES/'+path);sources.append(dict(path=path,sha256=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest(),LF=True))
    methods=[];refs={r['id']:r for r in staging['references']}
    for entry in runtime['results']:
        value=tick.load(closure.artifact(ROOT,entry));cid=value['id'];checkpoint=terminal['checkpoints'][1] if cid=='AdaRES-Y101' else dict(path=refs[cid]['mapped_checkpoint'],sha256=refs[cid]['expected_sha256'])
        methods.append(dict(id=cid,config=value['config'],config_sha256=value['config_sha256'],checkpoint=checkpoint['path'],checkpoint_sha256=checkpoint['sha256']))
    adapter_source=closure.artifact(ROOT,adapter['adapted_evaluator']).read_text(encoding='utf-8');analysis_source=(ROOT/'scripts/analyze_frozen_full.py').read_text(encoding='utf-8')
    folder='/data1/songcl/Project/AdaRES_project/pipeline/review_jobs/AdaRES-Y101/'+token
    p=dict(id='AdaRES-Y101',token=token,server='server-1',gpu=0,power=350.,old_pid=2263581,code=server['remote_code_root'],code_commit=Y101,candidate_config=c['config'],freeze=freeze,terminal=terminal,methods=methods,source_files=sources,folder=folder,python=server['python'],python_paths=runtime['launch_python_paths'],runtime_library_path=server['runtime_library_path'],adapter_source=adapter_source,adapter_sha256=hashlib.sha256(adapter_source.encode()).hexdigest(),strip_bootstrap=STRIP_BOOTSTRAP,analysis_source=analysis_source,analysis_sha256=hashlib.sha256(analysis_source.encode()).hexdigest())
    ns={'__name__':'validate_plan_only'};exec(compile(WORKER,'fixed_worker','exec'),ns);ns['validate'](p)
    program='P='+repr(p)+'\n'+WORKER;ast.parse(program,feature_version=(3,8))
    path=ROOT/('pipeline/checks/y101_review_execution_'+token+'.py');intent=ROOT/('pipeline/checks/y101_review_execution_'+token+'_intent.json');assert not path.exists() and not intent.exists();path.write_text(program,encoding='utf-8',newline='\n')
    tick.save(intent,dict(complete=True,token=token,id='AdaRES-Y101',status='launch_intent_no_PID_yet',program=ref(path),terminal=c['terminal_evidence'],archive=c['terminal_archive_binding'],freeze=c['post_training_freeze'],remote_folder=folder,scope='Fixed original four-method evaluation, not a scheduler/decision loop; no automatic retry/restart/promotion.',evidence=[ref(__file__),ref(path),c['frozen_review_worker_plan'],c['frozen_review_bootstrap_test'],c['frozen_review_resource_adapter'],c['frozen_review_runtime_binding'],c['terminal_evidence'],c['terminal_archive_binding']]))
    # A durable intent precedes remote launch; an uncertain result must be reconciled from this exact folder, never blindly rerun.
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert not c.get('frozen_review_job');c['frozen_review_launch_intent']=ref(intent);c['evidence'].extend([ref(intent),ref(path)]);tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    body=r'''import hashlib,json,os,pathlib,subprocess,sys
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
assert not pathlib.Path('/proc/2263581').exists()
root=pathlib.Path('/data1/songcl/Project/AdaRES_project').resolve();folder=pathlib.Path(Q['folder']);assert str(folder.resolve()).startswith(str(root)+'/pipeline/review_jobs/AdaRES-Y101/')
assert not folder.exists();folder.mkdir(parents=True);assert str(folder.resolve()).startswith(str(root)+'/pipeline/review_jobs/AdaRES-Y101/')
program=folder/'worker.py';raw=Q['program'].encode();assert hashlib.sha256(raw).hexdigest()==Q['program_sha256'];program.write_bytes(raw)
(folder/'launch_intent.json').write_text(json.dumps(dict(id='AdaRES-Y101',token=Q['token'],program_sha256=Q['program_sha256'],status='launching')))
environment=os.environ.copy();environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=Q['library'],PYTHONPATH=os.pathsep.join(Q['python_paths']))
with (folder/'worker.log').open('xb') as stream:
    child=subprocess.Popen([Q['python'],'-B','-S',str(program)],cwd=Q['code'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
proof=dict(complete=True,id='AdaRES-Y101',token=Q['token'],pid=child.pid,folder=str(folder),program_sha256=Q['program_sha256'],process_created=True,native_progress_confirmed=False)
(folder/'launch.json').write_text(json.dumps(proof));print(json.dumps(proof))
'''
    output=ROOT/('pipeline/checks/y101_review_execution_'+token+'_launch.json');value=remote(server,'Q='+repr(dict(folder=folder,token=token,program=program,program_sha256=ref(path)['sha256'],python=server['python'],library=server['runtime_library_path'],python_paths=p['python_paths'],code=server['remote_code_root']))+'\n'+body,output)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert all(ref(k)==v for k,v in immutable.items()) and git('ls-files','--stage','-z')==index
        work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');c['frozen_review_job']=dict(folder=folder,pid=value['pid'],token=token,launch=ref(output),program=ref(path),native_progress_confirmed=False);c['evidence'].extend([ref(output),ref(output.with_suffix('.log')),ref(__file__)]);c['next_action']='Observe this exact original-resource fixed four-method review job; retain partial/error evidence, never automatically retry or launch duplicate work. Finish paired/full cost/scientific review before any promotion. GPU1/Y57 holds preserved.';tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    print(json.dumps(dict(launch=ref(output),pid=value['pid'],native_progress_confirmed=False)))

def observe(token,label):
    lease(token,300);work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');job=c['frozen_review_job'];closure.artifact(ROOT,job['launch']);closure.artifact(ROOT,job['program'])
    server=monitor.load_servers(str(ROOT))['server-1'];out=ROOT/('pipeline/checks/y101_review_progress_'+token+'_'+label+'.json')
    body=r'''import json,pathlib,subprocess,time
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
folder=pathlib.Path(Q['folder']);assert folder.is_dir()
def read():
    state=json.loads((folder/'state.json').read_text());assert state['worker_pid']==Q['pid'] and state['token']==Q['job_token'];log=pathlib.Path(state['log']) if state.get('log') else None
    rows=[]
    if log and log.exists():
        for line in log.read_text(errors='replace').splitlines():
            try:r=json.loads(line)
            except (ValueError,TypeError):continue
            if isinstance(r,dict) and 'chunks' in r:rows.append(r)
    return dict(state=state,latest=rows[-1] if rows else None,worker_alive=pathlib.Path('/proc/'+str(Q['pid'])).is_dir())
a=read();time.sleep(5);b=read();compute=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip();pids=[int(x) for x in compute.splitlines() if x.strip()]
progress=(len(b['state'].get('completed_methods',[]))>len(a['state'].get('completed_methods',[]))) or (a['state'].get('child_pid')==b['state'].get('child_pid') and a['latest'] is not None and b['latest'] is not None and (b['latest']['pass'],b['latest']['chunks'])>(a['latest']['pass'],a['latest']['chunks']))
if b['state']['status']=='complete':progress=True
print(json.dumps(dict(complete=True,id='AdaRES-Y101',before=a,after=b,compute_pids=pids,native_progress_confirmed=progress,original_training_absent=not pathlib.Path('/proc/2263581').exists(),remote_mutations=False)))
'''
    value=remote(server,'Q='+repr(dict(folder=job['folder'],pid=job['pid'],job_token=job['token']))+'\n'+body,out)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');c['frozen_review_job']['latest_observation']=ref(out);c['frozen_review_job']['native_progress_confirmed']=value['native_progress_confirmed'] or c['frozen_review_job'].get('native_progress_confirmed',False);c['evidence'].extend([ref(out),ref(out.with_suffix('.log')),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    print(json.dumps(dict(observation=ref(out),state=value['after']['state'],latest=value['after']['latest'],native_progress_confirmed=value['native_progress_confirmed'],compute_pids=value['compute_pids'])))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('test','launch','observe'));parser.add_argument('--token',required=True);parser.add_argument('--label',default='start');a=parser.parse_args()
    if a.operation=='test':test(a.token)
    elif a.operation=='launch':launch(a.token)
    else:observe(a.token,a.label)
