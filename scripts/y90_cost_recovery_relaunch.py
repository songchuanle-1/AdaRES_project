"""Reconcile a refused resource-only attempt, then launch its unchanged guarded cost on the original GPU0."""
import argparse,ast,copy,hashlib,json
from datetime import datetime,timezone
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_frozen_review_job import remote
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import GOAL,PAUSE
from y102_deployment import git,Y102
import closure

RECONCILE=r'''import datetime,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
old=pathlib.Path(Q['refused_folder']);new=pathlib.Path(Q['new_folder'])
assert not old.exists() and not new.exists()
assert not pathlib.Path('/proc/3294283').exists()
pids=[int(x) for x in subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().splitlines() if x.strip()]
assert not pids
power=float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip());assert power==350.
print(json.dumps(dict(complete=True,token=Q['token'],observed_at=datetime.datetime.now().astimezone().isoformat(),hostname='dl02-MS-7B73',old_worker_absent=True,refused_folder_absent=True,refused_launch_receipt_absent=True,new_folder_absent=True,compute_pids=pids,power_limit_watts=power,remote_mutations=False,checkpoint_loaded=False,GPU_used=False)))
'''

LAUNCH=r'''import hashlib,json,os,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
assert not pathlib.Path('/proc/3294283').exists() and not pathlib.Path(Q['refused_folder']).exists()
assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
assert float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip())==350.
folder=pathlib.Path(Q['folder']);assert folder.parent==pathlib.Path('/data1/songcl/Project/AdaRES_project/pipeline/cost_jobs/AdaRES-Y90') and not folder.exists();folder.mkdir(parents=True)
raw=Q['program'].encode();assert hashlib.sha256(raw).hexdigest()==Q['sha'];path=folder/'worker.py';path.write_bytes(raw);(folder/'launch_intent.json').write_text(json.dumps(dict(token=Q['token'],program_sha256=Q['sha'],attempt=3)))
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=Q['library'],PYTHONPATH=os.pathsep.join(Q['paths']))
with (folder/'worker.log').open('xb') as stream:child=subprocess.Popen([Q['python'],'-B','-S',str(path)],cwd=Q['root'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
v=dict(complete=True,id='AdaRES-Y90',token=Q['token'],pid=child.pid,folder=str(folder),program_sha256=Q['sha'],process_created=True,native_progress_confirmed=False,attempt=3);(folder/'launch.json').write_text(json.dumps(v));print(json.dumps(v))
'''

def launch(token):
    lease(token,900)
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['preflight']['goal_rc']==4 and view['queue']['candidate_budget']==view['research']['creation_budget']==0
    protected={p:ref(p) for p in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/experiments.csv','pipeline/admissions/AdaRES-Y90.json','pipeline/admissions/AdaRES-Y102.json')};index=git('ls-files','--stage','-z')
    assert protected['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE and git('rev-parse','HEAD').decode().strip()==Y102
    w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y90')
    assert c['stage']=='validation' and c['cost_resource_recovery_status']=='guard_refused_no_remote_job_created' and c['frozen_trained_cost_job']['pid']==3294283 and not c.get('complete_scientific_review') and not c.get('cost_resource_recovery_relaunch_intent')
    hold=tick.load(closure.artifact(ROOT,c['cost_validation_hold']));assert hold['launch_uncertainty_resolved'] and not hold['GPU_job_started']
    original_plan_ref=copy.deepcopy(c['cost_resource_recovery_plan']);plan=tick.load(closure.artifact(ROOT,original_plan_ref));assert plan['original_scientific_core_byte_identical'] and plan['safety_guard_boundary_checks_passed']==8 and not plan['GPU_started']
    old=closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8');p=ast.literal_eval(ast.parse(old).body[0].value);assert p['gpu']==0 and p['power']==350. and hashlib.sha256(p['core'].encode()).hexdigest()==p['core_sha256']
    new=copy.deepcopy(p);new.update(token=token,folder=p['folder'].rsplit('/',1)[0]+'/'+token+'_resource_recovery',output=p['folder'].rsplit('/',1)[0]+'/'+token+'_resource_recovery/trained_cost.json')
    assert {k for k in p if p[k]!=new[k]}=={'token','folder','output'}
    code='P='+repr(new)+'\n'+old.split('\n',1)[1];ast.parse(code,feature_version=(3,8));assert code.split('\n',1)[1]==old.split('\n',1)[1]
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0]
    q=monitor.load_queue(str(ROOT));assert not monitor.load_running(str(ROOT)).get('server-1',{}).get('0') and not any(x['status'] in ('running','claiming') for x in q['gpu_queues']['server-1'])
    # Canonical idle collection already unblocked every matching hold; use fresh remote reconciliation below.
    occupation=ref('pipeline/checks/gpu_occupation_server-1_start_'+token+'.json');o=tick.load(closure.artifact(ROOT,occupation));assert o['complete'] and o['scope']['allowed_gpus']==[0] and not o['queries']['compute']['stdout'].strip()
    reconciliation=ROOT/('pipeline/checks/y90_cost_relaunch_reconciliation_'+token+'.json');remote(server,'Q='+repr(dict(token=token,refused_folder=p['folder'],new_folder=new['folder']))+'\n'+RECONCILE,reconciliation)
    program=ROOT/('pipeline/checks/y90_cost_guarded_relaunch_'+token+'.py');newplan=program.with_suffix('.json');intent=ROOT/('pipeline/checks/y90_cost_guarded_relaunch_intent_'+token+'.json')
    assert not program.exists() and not newplan.exists() and not intent.exists();program.write_text(code,encoding='utf-8',newline='\n')
    tick.save(newplan,dict(complete=True,id='AdaRES-Y90',token=token,attempt=3,bounded_resource_recovery=True,original_preparation_token=plan['token'],original_prepared_plan=original_plan_ref,original_guarded_program=plan['program'],program=ref(program),only_payload_metadata_changed=['token','folder','output'],worker_body_byte_identical=True,original_scientific_core_byte_identical=True,original_core_sha256=p['core_sha256'],original_imports_capture_checkpoint_source_bindings_identical=True,prior_guard_tests_reused_not_replayed=8,original_refused_intent=c['cost_resource_recovery_launch_intent'],resolved_previous_attempt=c['cost_validation_hold'],reconciliation=ref(reconciliation),occupation=occupation,GPU_started=False,checkpoint_loaded=False,evidence=[ref(__file__),original_plan_ref,ref(reconciliation),occupation,ref(program)]))
    tick.save(intent,dict(complete=True,token=token,id='AdaRES-Y90',status='intent_before_remote_creation',attempt=3,program=ref(program),plan=ref(newplan),folder=new['folder'],previous_intent_preserved=c['cost_resource_recovery_launch_intent'],previous_uncertainty_resolved=c['cost_validation_hold']))
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y90');assert not c.get('cost_resource_recovery_relaunch_intent');c['cost_resource_recovery_relaunch_intent']=ref(intent);c['evidence'].extend([ref(newplan),ref(intent),ref(reconciliation),occupation,ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:lock.close()
    out=ROOT/('pipeline/checks/y90_cost_guarded_relaunch_launch_'+token+'.json')
    value=remote(server,'Q='+repr(dict(folder=new['folder'],refused_folder=p['folder'],program=code,sha=ref(program)['sha256'],token=token,python=server['python'],library=server['runtime_library_path'],paths=new['python_paths'],root=server['remote_code_root']))+'\n'+LAUNCH,out)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y90');assert c['frozen_trained_cost_job']['pid']==3294283
        job=dict(folder=new['folder'],token=token,pid=value['pid'],launch=ref(out),program=ref(program),native_progress_confirmed=False,attempt=3)
        c['frozen_trained_cost_job']=job;c['frozen_trained_cost_plan']=ref(newplan);c['frozen_trained_cost_launch_intent']=ref(intent);c['cost_resource_recovery_status']='running_awaiting_guarded_cost_postaudit';c.setdefault('frozen_trained_cost_attempts',[]).append(dict(attempt=3,status='process_created_progress_not_yet_confirmed',job=copy.deepcopy(job),plan=ref(newplan),intent=ref(intent),scientifically_valid_cost=False))
        c['evidence'].extend([ref(out),ref(out.with_suffix('.log'))]);w['tick_attempt']=dict(tick_started_at=view['generated_at'],candidate_id='AdaRES-Y90',kind='resource_check',result='passed',command='python -B scripts/y90_cost_recovery_relaunch.py --token '+token,observation='Original server1GPU0 freshly exclusive idle; exact prior refused folder absent and old worker absent; unchanged guarded frozen-cost worker process created, progress pending.',evidence=[ref(newplan),ref(reconciliation),occupation,ref(out)])
        tick.save(ROOT/'pipeline/research_work.json',w);assert all(ref(p)==r for p,r in protected.items()) and git('ls-files','--stage','-z')==index
    finally:lock.close()
    print(json.dumps(dict(launch=ref(out),plan=ref(newplan),pid=value['pid'],attempt=3,scientific_core_and_worker_body_byte_identical=True,native_progress_confirmed=False,new_training_launches=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);launch(parser.parse_args().token)
