"""Launch only the missing three Y88 baselines after a freshly exclusive original GPU0 slot is available."""
import argparse,ast,copy,json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_frozen_review_job import remote
import closure

BODY=r'''import hashlib,json,os,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
old=pathlib.Path(Q['old_folder']);state=json.loads((old/'state.json').read_text());assert state['status']=='failed' and state['completed_methods']==['AdaRES-Y88']
assert not pathlib.Path('/proc/'+str(Q['old_pid'])).exists()
assert hashlib.sha256((old/'worker.py').read_bytes()).hexdigest()==Q['old_program_sha256']
assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
power=float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip());assert power==350.
used,util=[float(v.strip()) for v in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')];assert used<1024 and util<10
folder=pathlib.Path(Q['folder']);assert folder.parent==pathlib.Path('/data1/songcl/Project/AdaRES_project/pipeline/review_jobs/AdaRES-Y88') and not folder.exists();folder.mkdir(parents=True)
raw=Q['program'].encode();assert hashlib.sha256(raw).hexdigest()==Q['sha'];path=folder/'worker.py';path.write_bytes(raw);(folder/'launch_intent.json').write_text(json.dumps(dict(token=Q['token'],program_sha256=Q['sha'],status='launching_remaining_only')))
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=Q['library'],PYTHONPATH=os.pathsep.join(Q['paths']))
with (folder/'worker.log').open('xb') as stream:child=subprocess.Popen([Q['python'],'-B','-S',str(path)],cwd=Q['root'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
v=dict(complete=True,id='AdaRES-Y88',token=Q['token'],pid=child.pid,folder=str(folder),program_sha256=Q['sha'],process_created=True,native_progress_confirmed=False,attempt=2,reused_methods=['AdaRES-Y88'],new_methods=['AdaRES-Y36','AdaRES-W2','AdaRES-B1']);(folder/'launch.json').write_text(json.dumps(v));print(json.dumps(v))
'''

def launch(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y88');assert c['stage'] in ('blocked','validation') and not c.get('full_frozen_evaluation_result') and not c.get('frozen_review_continuation_launch_intent')
    if c['stage']=='blocked':assert c['blocker_kind']=='exclusive_gpu_slot' and c['required_server']=='server-1' and c['required_gpu']==0
    old=copy.deepcopy(c['frozen_review_job']);assert old['failed'] and old['worker_absent'] and old['completed_methods']==['AdaRES-Y88'] and len(c['frozen_review_attempts'])==1
    plan_ref=c['frozen_review_continuation_plan'];plan=tick.load(closure.artifact(ROOT,plan_ref));assert plan['completed_capture_identity_coverage_boundary_checks_passed']==8 and plan['native_evaluator_adapter_bootstrap_analysis_payload_byte_identical'] and not plan['launched']
    original=closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8');p=ast.literal_eval(ast.parse(original).body[0].value);fresh=copy.deepcopy(p);fresh.update(token=token,folder=p['folder'].rsplit('/',1)[0]+'/'+token+'_remaining')
    assert all(p[k]==v for k,v in fresh.items() if k not in ('token','folder')) and list(p['resume_captures'])==['AdaRES-Y88'];source='P='+repr(fresh)+'\n'+original.split('\n',1)[1];ast.parse(source,feature_version=(3,8))
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0];q=monitor.load_queue(str(ROOT));assert not monitor.load_running(str(ROOT)).get('server-1',{}).get('0') and not any(r['status'] in ('running','claiming') for r in q['gpu_queues']['server-1'])
    program=ROOT/('pipeline/checks/y88_remaining_launch_program_'+token+'.py');binding=program.with_suffix('.json');intent=ROOT/('pipeline/checks/y88_remaining_launch_intent_'+token+'.json');assert not program.exists() and not binding.exists() and not intent.exists();program.write_text(source,encoding='utf-8',newline='\n')
    tick.save(binding,dict(complete=True,id='AdaRES-Y88',token=token,original_preparation_token=plan['token'],prepared_plan=plan_ref,program=ref(program),only_current_metadata_rebound=True,worker_body_byte_identical=True,scientific_payload_byte_identical=True,completed_candidate_capture_reused_not_replayed=True,evidence=[plan_ref,ref(program),ref(__file__)]))
    tick.save(intent,dict(complete=True,id='AdaRES-Y88',token=token,status='intent_before_remote_creation',prepared_plan=plan_ref,current_binding=ref(binding),program=ref(program),folder=fresh['folder'],original_failed_job=old,attempt=2))
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y88');assert not c.get('frozen_review_continuation_launch_intent');c['frozen_review_continuation_launch_intent']=ref(intent);c['evidence'].extend([ref(intent),ref(binding),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:lock.close()
    out=ROOT/('pipeline/checks/y88_remaining_launch_'+token+'.json');result=remote(server,'Q='+repr(dict(old_folder=old['folder'],old_pid=old['pid'],old_program_sha256=old['program']['sha256'],folder=fresh['folder'],program=source,sha=ref(program)['sha256'],token=token,python=server['python'],library=server['runtime_library_path'],paths=fresh['python_paths'],root=server['remote_code_root']))+'\n'+BODY,out)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y88');assert c['frozen_review_job']==old and len(c['frozen_review_attempts'])==1
        job=dict(folder=fresh['folder'],token=token,pid=result['pid'],launch=ref(out),program=ref(program),native_progress_confirmed=False,attempt=2,reused_completed_methods=['AdaRES-Y88'])
        c['frozen_review_attempts'].append(dict(attempt=2,job=copy.deepcopy(job),intent=ref(intent),binding=ref(binding),status='process_created_progress_pending'));c.update(stage='validation',frozen_review_job=job,frozen_review_launch_intent=ref(intent),frozen_review_continuation_launch_binding=ref(binding),scientific_status='candidate_capture_reused_remaining_baselines_running',next_action='Observe exact original server1GPU0/350W continuation. Existing candidate two-pass output is reused byte-for-byte; only Y36/W2/B1 need original two passes. Preserve old failed worker and all raw evidence. Reconcile all eight captures/full original comparisons, then original-device cost/postaudit; no duplicate launch, completed candidate replay, training restart or automatic scientific promotion.')
        for key in ('blocker','blocker_kind','blocked_from_stage','retry_after','unblock_condition','occupation_evidence'):c.pop(key,None)
        c['evidence'].extend([ref(out),ref(out.with_suffix('.log'))]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:lock.close()
    print(json.dumps(dict(launch=ref(out),pid=result['pid'],native_progress_confirmed=False,reused_candidate_predictions=True,new_training=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);launch(parser.parse_args().token)
