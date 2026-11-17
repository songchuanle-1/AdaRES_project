"""Record actual remaining-only Y88 progress and hold other original GPU0 work using live evidence."""
import argparse,ast,json
from datetime import datetime,timedelta,timezone
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_frozen_review_job import remote
import closure

BODY=r'''import datetime,hashlib,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
folder=pathlib.Path(Q['folder']);state=json.loads((folder/'state.json').read_text());assert state['worker_pid']==Q['pid'] and state['token']==Q['token'] and state['status']=='evaluating' and state['method'] in ('AdaRES-Y36','AdaRES-W2','AdaRES-B1') and state['completed_methods'][0]=='AdaRES-Y88'
assert hashlib.sha256((folder/'worker.py').read_bytes()).hexdigest()==Q['worker_sha256']
rows=[]
for name,digest in (('adares-y88.json',Q['capture_sha256']),('adares-y88.log',Q['log_sha256'])):
 path=folder/name;raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==digest;rows.append(dict(name=name,bytes=len(raw),sha256=digest))
pids=[int(x) for x in subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().splitlines() if x.strip()]
print(json.dumps(dict(complete=True,id='AdaRES-Y88',token=Q['token'],observed_at=datetime.datetime.now().astimezone().isoformat(),state=state,compute_pids=pids,child_compute_observed=state['child_pid'] in pids,reused_files=rows,completed_candidate_replayed=False,remote_mutations=False,checkpoint_loaded=False)))
'''

def record(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in w['candidates']};y=c['AdaRES-Y88'];job=y['frozen_review_job'];obsref=job['latest_observation'];obs=tick.load(closure.artifact(ROOT,obsref));assert job['attempt']==2 and job['token']==token and obs['native_progress_confirmed'] and obs['after']['worker_alive']
    plan=tick.load(closure.artifact(ROOT,y['frozen_review_continuation_plan']));prepared=closure.artifact(ROOT,plan['program']).read_text(encoding='utf8');actual=closure.artifact(ROOT,job['program']).read_text(encoding='utf8');p=ast.literal_eval(ast.parse(actual).body[0].value);old=ast.literal_eval(ast.parse(prepared).body[0].value);assert actual.split('\n',1)[1]==prepared.split('\n',1)[1] and {k for k in old if old[k]!=p[k]}=={'token','folder'}
    entry=p['resume_captures']['AdaRES-Y88'];out=ROOT/('pipeline/checks/y88_resume_reuse_verification_'+token+'_active.json');value=remote(monitor.load_servers(str(ROOT))['server-1'],'Q='+repr(dict(folder=job['folder'],pid=job['pid'],token=token,worker_sha256=job['program']['sha256'],capture_sha256=entry['capture_sha256'],log_sha256=entry['log_sha256']))+'\n'+BODY,out);assert value['child_compute_observed'],'Preserve the read-only snapshot; use actual active phase, never guess occupancy'
    failed=ref('pipeline/checks/y88_resume_reuse_verification_'+token+'.log');assert 'line 9' in closure.artifact(ROOT,failed).read_text(encoding='utf8')
    correction=ROOT/('pipeline/checks/y88_method_transition_observation_'+token+'.json');assert not correction.exists();tick.save(correction,dict(complete=True,token=token,classification='read_only_compute_membership_snapshot_failure_during_method_transition',failed_probe=failed,failed_scientific_test=False,launch_uncertainty=False,remote_mutations=False,original_worker_preserved=True,original_observer_confirmed_continuing_progress=obsref,current_active_snapshot=ref(out),completed_candidate_byte_identity_verified=True,evidence=[failed,obsref,ref(out),ref(__file__)]))
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in w['candidates']};y=c['AdaRES-Y88'];y.update(current_review_worker_alive=True,current_native_progress_confirmed=True,continuation_reused_capture_verification=ref(out),full_review_collection_entry=ref('scripts/collect_y88_remaining_full_review.py'),guarded_frozen_cost_entry=ref('scripts/y88_guarded_trained_cost.py'));y['evidence'].extend([ref(out),ref(correction),ref(__file__),y['full_review_collection_entry'],y['guarded_frozen_cost_entry']])
        ids=tick.load(ROOT/'pipeline/headless/.tick_context.json')['research_work']['due_ids'];waiting=[c[i] for i in ids if i!='AdaRES-Y88'];assert len(waiting)==11
        for x in waiting:
            assert x['stage']=='validation' and x['required_server']=='server-1';x.update(stage='blocked',blocked_from_stage='validation',blocker_kind='exclusive_gpu_slot',blocker='Original server1GPU0/350W is occupied by the verified Y88 remaining-baselines review. Preserve its live worker/child and all original-device requirements.',retry_after=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),unblock_condition='Fresh exclusive idle/unreserved original server1GPU0/350W after existing frozen work releases it; no GPU1 substitution, foreign-job interference, new training under zero phase budget or blind replay.');x['evidence'].append(ref(out))
        tick.save(ROOT/'pipeline/research_work.json',w)
    finally:lock.close()
    print(json.dumps(dict(reuse_verification=ref(out),method_transition_observation=ref(correction),state=value['state'],compute_pids=value['compute_pids'],completed_candidate_predictions_replayed=False,other_original_GPU0_holds=11)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);record(parser.parse_args().token)
