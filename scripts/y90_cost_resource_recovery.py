"""Preserve a resource-confounded cost attempt and permit one same-protocol monitored recovery."""
import argparse,ast,copy,hashlib,json,subprocess
from datetime import datetime,timezone
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_frozen_review_job import remote
from complete_y101_scientific_review import BODY as COLLECT
import closure

GUARD=r"""import threading
GUARD_STOP=threading.Event();GUARD_FAILED=threading.Event();GUARD_LOCK=threading.Lock();GUARD_SAMPLES=0;GUARD_OWNED_SAMPLES=0

def allowed_compute(pids,own):
    if not isinstance(pids,(list,tuple)) or not isinstance(own,int) or own<=0:raise ValueError('Unknown compute inventory')
    if any(not isinstance(pid,int) or isinstance(pid,bool) or pid<=0 for pid in pids):raise ValueError('Malformed compute inventory')
    if any(pid!=own for pid in pids):raise ValueError('Unowned GPU0 compute observed')
    return True

def resource_sample():
    global GUARD_SAMPLES,GUARD_OWNED_SAMPLES
    row=dict(observed_at=datetime.datetime.now().astimezone().isoformat(),own_pid=os.getpid())
    try:
        result=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=3)
        row.update(rc=result.returncode,stdout=result.stdout,stderr=result.stderr)
        if result.returncode!=0:raise ValueError('Compute query failed')
        pids=[int(line.strip()) for line in result.stdout.splitlines() if line.strip()];row['compute_pids']=pids;allowed_compute(pids,os.getpid());row['passed']=True
    except BaseException as error:row.update(passed=False,error=repr(error));GUARD_FAILED.set()
    with GUARD_LOCK:
        GUARD_SAMPLES+=1
        if os.getpid() in row.get('compute_pids',[]):GUARD_OWNED_SAMPLES+=1
        with (folder/'resource_guard.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush()
    return row

def resource_check():
    if GUARD_FAILED.is_set():raise RuntimeError('GPU0 resource guard failed; preserve raw cost and resource log, no valid cost disposition')

def resource_watch():
    while not GUARD_STOP.wait(1):resource_sample()
"""

def prepare(token):
    lease(token,900);w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y90');assert c['stage']=='validation' and not c.get('complete_scientific_review') and not c.get('cost_resource_recovery_plan')
    job=copy.deepcopy(c['frozen_trained_cost_job']);old_program=closure.artifact(ROOT,job['program']);old_text=old_program.read_text(encoding='utf-8');old_p=ast.literal_eval(ast.parse(old_text).body[0].value);assert job['pid']==3294283 and old_p['id']=='AdaRES-Y90' and old_p['gpu']==0 and old_p['power']==350. and old_p['token']==token
    evidence=ref('pipeline/checks/y90_cost_progress_'+token+'_start.json');observed=tick.load(closure.artifact(ROOT,evidence));assert observed['compute_pids']==[3294544,3294283] or set(observed['compute_pids'])=={3294544,3294283}
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0]
    archive=ROOT/('pipeline/analyses/y90_cost_resource_contended_'+token);assert not archive.exists();archive.mkdir();meta=archive/'remote_manifest.json'
    v=remote(server,'Q='+repr(dict(folder=job['folder'],pid=job['pid'],program_sha256=ref(old_program)['sha256']))+'\n'+COLLECT.replace('AdaRES-Y101','AdaRES-Y90'),meta)
    for row in v['files']:
        target=archive/row['name'];assert not target.exists() and target.resolve().is_relative_to(archive.resolve());subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',server['user']+'@'+server['host']+':'+row['path'],str(target)],check=True,timeout=90);assert target.stat().st_size==row['bytes'] and ref(target)['sha256']==row['sha256']
    raw_audit=tick.load(archive/'postaudit.json');assert raw_audit['complete'] and raw_audit['source_raw_bytes_preserved'] and raw_audit['cost_sha256']==ref(archive/'trained_cost.json')['sha256']
    invalid=archive/'resource_disposition.json';tick.save(invalid,dict(complete=True,id='AdaRES-Y90',token=token,scientifically_valid_cost=False,reason='Extra GPU0 compute PID3294544 observed concurrently with original cost worker3294283; extra identity could not be established after it disappeared. Preserve all raw measurements and successful source/checkpoint audit; no causal or isolated timing claim.',original_job=job,original_plan=c['frozen_trained_cost_plan'],original_intent=c['frozen_trained_cost_launch_intent'],raw_archive=archive.relative_to(ROOT).as_posix(),formal_predictions_affected=False,original_weights_unchanged=True,evidence=[evidence,ref(meta),ref(archive/'trained_cost.json'),ref(archive/'postaudit.json'),ref('pipeline/checks/y90_cost_extra_compute_'+token+'.json'),ref('pipeline/checks/server1_new_compute_'+token+'.json'),ref(__file__)]))
    payload=copy.deepcopy(old_p);payload['folder']=old_p['folder']+'_resource_recovery1';payload['output']=payload['folder']+'/trained_cost.json';payload['original_contended_attempt']=ref(invalid)
    assert payload['core']==old_p['core'] and payload['core_sha256']==old_p['core_sha256'] and hashlib.sha256(payload['core'].encode()).hexdigest()==payload['core_sha256'] and payload['capture_files']==old_p['capture_files'] and payload['checkpoint_audit']==old_p['checkpoint_audit']
    header=old_text.split('\n',1)[1];assert header.count('COUNT=0\n')==1;header=header.replace('COUNT=0\n','COUNT=0\n'+GUARD+'\n')
    needle='    COUNT+=1;value=';assert header.count(needle)==1;header=header.replace(needle,'    resource_check()\n    COUNT+=1;value=')
    needle="    root=pathlib.Path(P['root']);source_manifest=P['candidate_source_manifest']";assert header.count(needle)==1;header=header.replace(needle,"    resource_sample();resource_check();GUARD_THREAD=threading.Thread(target=resource_watch,daemon=True);GUARD_THREAD.start()\n"+needle)
    needle="    audit=dict(complete=True,id='AdaRES-Y90'";assert header.count(needle)==1;header=header.replace(needle,"    GUARD_STOP.set();GUARD_THREAD.join(5);assert not GUARD_THREAD.is_alive();resource_sample();resource_check();assert GUARD_SAMPLES>=4 and GUARD_OWNED_SAMPLES>0\n"+needle)
    needle='source_raw_bytes_preserved=True,source_files=after_sources';assert header.count(needle)==1;header=header.replace(needle,"source_raw_bytes_preserved=True,resource_guard_samples_complete=True,resource_guard_samples=GUARD_SAMPLES,resource_guard_owned_samples=GUARD_OWNED_SAMPLES,no_unowned_GPU0_compute_observed=True,resource_guard_log_sha256=sha(folder/'resource_guard.jsonl'),source_files=after_sources")
    code='P='+repr(payload)+'\n'+header;ast.parse(code,feature_version=(3,8));guard_tree=ast.parse(GUARD);function=next(n for n in guard_tree.body if isinstance(n,ast.FunctionDef) and n.name=='allowed_compute');ns={};exec(compile(ast.Module(body=[function],type_ignores=[]),'resource_guard_tests','exec'),ns);fn=ns['allowed_compute'];assert fn([],123) and fn([123],123) and fn([123,123],123)
    for values in ([124],[123,124],[0],['123'],None):
        try:fn(values,123)
        except ValueError:pass
        else:raise AssertionError('Accepted unsafe compute inventory')
    base='pipeline/checks/y90_cost_resource_recovery_'+token;program=ROOT/(base+'.py');plan=ROOT/(base+'.json');assert not program.exists() and not plan.exists();program.write_text(code,encoding='utf-8',newline='\n')
    tick.save(plan,dict(complete=True,id='AdaRES-Y90',token=token,attempt=2,bounded_resource_recovery=True,program=ref(program),original_contended_attempt=ref(invalid),original_cost_program=job['program'],original_scientific_core_byte_identical=True,original_core_sha256=old_p['core_sha256'],original_imports_and_capture_checkpoint_bindings_byte_identical=True,only_resource_instrumentation_outside_scientific_core=True,safety_guard_boundary_checks_passed=8,sampling_seconds=1,automation_schedule_unchanged=True,guard_lifetime='Only this fixed cost worker; no project polling/decisions or foreign-process mutations',GPU_started=False,checkpoint_loaded=False,evidence=[ref(__file__),ref(program),ref(invalid),job['program'],c['post_training_freeze'],c['full_frozen_evaluation_result']]))
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y90');assert not c.get('cost_resource_recovery_plan');c.setdefault('frozen_trained_cost_attempts',[]).append(dict(attempt=1,job=job,plan=c['frozen_trained_cost_plan'],intent=c['frozen_trained_cost_launch_intent'],disposition=ref(invalid),scientifically_valid_cost=False));c['cost_resource_recovery_plan']=ref(plan);c['evidence'].extend([ref(invalid),ref(plan),ref(meta),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:lock.close()
    print(json.dumps(dict(plan=ref(plan),invalid_first_attempt=ref(invalid),scientific_core_byte_identical=True,resource_guard_tests=8,GPU_started=False)))

def launch(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y90');assert c['stage']=='validation' and c['frozen_trained_cost_job']['pid']==3294283 and not c.get('cost_resource_recovery_launch_intent')
    plan=tick.load(closure.artifact(ROOT,c['cost_resource_recovery_plan']));program=closure.artifact(ROOT,plan['program']);text=program.read_text(encoding='utf-8');p=ast.literal_eval(ast.parse(text).body[0].value);assert plan['bounded_resource_recovery'] and plan['original_scientific_core_byte_identical'] and p['gpu']==0 and p['power']==350.
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0];q=monitor.load_queue(str(ROOT));assert not monitor.load_running(str(ROOT)).get('server-1',{}).get('0') and not any(x['status'] in ('running','claiming') for x in q['gpu_queues']['server-1'])
    intent=ROOT/('pipeline/checks/y90_cost_resource_recovery_intent_'+token+'.json');assert not intent.exists();tick.save(intent,dict(complete=True,token=token,id='AdaRES-Y90',status='intent_before_remote_creation',program=ref(program),plan=c['cost_resource_recovery_plan'],folder=p['folder'],bounded_attempt=2))
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y90');c['cost_resource_recovery_launch_intent']=ref(intent);c['evidence'].append(ref(intent));tick.save(ROOT/'pipeline/research_work.json',w)
    finally:lock.close()
    body=r"""import hashlib,json,os,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
assert not pathlib.Path('/proc/3294283').exists()
assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
assert float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip())==350.
folder=pathlib.Path(Q['folder']);assert str(folder.resolve()).startswith('/data1/songcl/Project/AdaRES_project/pipeline/cost_jobs/AdaRES-Y90/') and not folder.exists();folder.mkdir(parents=True)
raw=Q['program'].encode();assert hashlib.sha256(raw).hexdigest()==Q['sha'];path=folder/'worker.py';path.write_bytes(raw);(folder/'launch_intent.json').write_text(json.dumps(dict(token=Q['token'],program_sha256=Q['sha'])))
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=Q['library'],PYTHONPATH=os.pathsep.join(Q['paths']))
with (folder/'worker.log').open('xb') as stream:child=subprocess.Popen([Q['python'],'-B','-S',str(path)],cwd=Q['root'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
v=dict(complete=True,id='AdaRES-Y90',token=Q['token'],pid=child.pid,folder=str(folder),program_sha256=Q['sha'],process_created=True,native_progress_confirmed=False,attempt=2);(folder/'launch.json').write_text(json.dumps(v));print(json.dumps(v))
"""
    out=ROOT/('pipeline/checks/y90_cost_resource_recovery_launch_'+token+'.json');v=remote(server,'Q='+repr(dict(folder=p['folder'],program=text,sha=ref(program)['sha256'],token=token,python=server['python'],library=server['runtime_library_path'],paths=p['python_paths'],root=server['remote_code_root']))+'\n'+body,out)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y90');assert c['frozen_trained_cost_job']['pid']==3294283;c['frozen_trained_cost_job']=dict(folder=p['folder'],token=token,pid=v['pid'],launch=ref(out),program=ref(program),native_progress_confirmed=False,attempt=2);c['frozen_trained_cost_plan']=c['cost_resource_recovery_plan'];c['frozen_trained_cost_launch_intent']=ref(intent);c['evidence'].extend([ref(out),ref(out.with_suffix('.log')),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:lock.close()
    print(json.dumps(dict(launch=ref(out),pid=v['pid'],attempt=2,scientific_core_byte_identical=True,native_progress_confirmed=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('prepare','launch'));parser.add_argument('--token',required=True);a=parser.parse_args();globals()[a.operation](a.token)
