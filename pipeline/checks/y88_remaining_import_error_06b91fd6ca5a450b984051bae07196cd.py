"""Preserve the interrupted Y88 review and prepare a guarded continuation of only its missing methods."""
import argparse,ast,copy,hashlib,json,math,subprocess
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_frozen_review_job import remote
from analyze_frozen_full import keyed,summarize
import closure

RECONCILE=r'''import datetime,hashlib,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
folder=pathlib.Path(Q['folder']);assert folder.parent==pathlib.Path('/data1/songcl/Project/AdaRES_project/pipeline/review_jobs/AdaRES-Y88')
state=json.loads((folder/'state.json').read_text());launch=json.loads((folder/'launch.json').read_text())
assert state['status']=='failed' and state['completed_methods']==['AdaRES-Y88'] and state['worker_pid']==launch['pid']==Q['pid'] and state['token']==launch['token']==Q['job_token']
assert not pathlib.Path('/proc/'+str(Q['pid'])).exists() and not pathlib.Path('/proc/3891626').exists()
assert hashlib.sha256((folder/'worker.py').read_bytes()).hexdigest()==Q['program_sha256']
log=(folder/'worker.log').read_text();assert 'in idle' in log and '--query-compute-apps=pid' in log and 'AssertionError' in log
for name in ('adares-y36','adares-w2','adares-b1'):assert not (folder/(name+'.json')).exists() and not (folder/(name+'.log')).exists()
files=[]
for name in ('state.json','launch.json','launch_intent.json','worker.log','adares-y88.json','adares-y88.log'):
    path=folder/name;a=path.stat();digest=hashlib.sha256(path.read_bytes()).hexdigest();b=path.stat();assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns);files.append(dict(name=name,path=str(path),bytes=a.st_size,sha256=digest))
result=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,universal_newlines=True,timeout=10);assert result.returncode==0
pids=[int(x) for x in result.stdout.splitlines() if x.strip()];processes=[]
for pid in pids:
    p=pathlib.Path('/proc')/str(pid);row=dict(pid=pid,exists=p.exists())
    if p.exists():row.update(status=[s for s in (p/'status').read_text().splitlines() if s.startswith(('Name:','State:','PPid:','Uid:'))],commandline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
    processes.append(row)
print(json.dumps(dict(complete=True,id='AdaRES-Y88',observed_at=datetime.datetime.now().astimezone().isoformat(),state=state,launch=launch,old_worker_and_child_absent=True,missing_baselines_never_started=True,files=files,compute_pids=pids,processes=processes,checkpoint_loaded=False,GPU_used=False,remote_mutations=False)))
'''

REUSE=r'''
def reuse_completed(p,folder):
    assert list(p['resume_captures'])==['AdaRES-Y88'] and [r['id'] for r in p['methods']]==IDS
    entry=p['resume_captures']['AdaRES-Y88'];row=p['methods'][0];assert entry['method']==row
    assert pathlib.Path(entry['capture_path']).parent==pathlib.Path(entry['log_path']).parent and str(pathlib.Path(entry['capture_path']).parent).startswith('/data1/songcl/Project/AdaRES_project/pipeline/review_jobs/AdaRES-Y88/')
    assert sha(entry['capture_path'])==entry['capture_sha256'] and sha(entry['log_path'])==entry['log_sha256']
    capture=json.loads(pathlib.Path(entry['capture_path']).read_text());assert capture['complete'] and capture['completed_passes']==2 and len(capture['passes'])==2 and capture['seed']==1999 and capture['size_cuts_from_training']==[1487,3325]
    assert all(capture[k]==row[k] for k in ('config','config_sha256','checkpoint','checkpoint_sha256'))
    assert capture['active_queries']==[128] and capture['selected_queries']==1 and all(v['summary']['expressions']==9508 and v['summary']['scenes']==141 and len(v['rows'])==9508 for v in capture['passes'])
    assert sha(row['config'])==row['config_sha256'] and sha(row['checkpoint'])==row['checkpoint_sha256']
    for source,name,digest in ((entry['capture_path'],'adares-y88.json',entry['capture_sha256']),(entry['log_path'],'adares-y88.log',entry['log_sha256'])):
        target=folder/name;assert not target.exists();raw=pathlib.Path(source).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest;target.write_bytes(raw);assert sha(target)==digest
    return {'AdaRES-Y88':capture}
'''

def validate_capture(capture,row):
    assert capture['complete'] and capture['completed_passes']==2 and len(capture['passes'])==2 and capture['seed']==1999 and capture['size_cuts_from_training']==[1487,3325]
    assert all(capture[k]==row[k] for k in ('config','config_sha256','checkpoint','checkpoint_sha256'))
    assert capture['active_queries']==[128] and capture['selected_queries']==1
    for each in capture['passes']:
        rows=each['rows'];assert len(rows)==len(keyed(rows))==9508 and len({r['scan_id'] for r in rows})==141
        assert all(math.isfinite(r['piou']) and 0<=r['piou']<=1 and math.isfinite(r['spiou']) and 0<=r['spiou']<=1 and r['size']==('small' if r['target_points']<=1487 else 'medium' if r['target_points']<=3325 else 'large') for r in rows)
        assert all(each['summary'][k]==v for k,v in summarize(rows).items())
    first,second=[v['rows'] for v in capture['passes']];assert list(keyed(first))==list(keyed(second));assert max(abs(a['piou']-b['piou']) for a,b in zip(first,second))==capture['repeat_max_iou_difference']
    return True

def prepare(token):
    lease(token,900);w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y88');assert c['stage']=='validation' and not c.get('frozen_review_continuation_plan') and not c.get('full_frozen_evaluation_result')
    job=copy.deepcopy(c['frozen_review_job']);assert job['pid']==3891354 and job['token']=='5a320b25043948e2bae2ffdef63da17a';old=closure.artifact(ROOT,job['program']).read_text(encoding='utf-8');p=ast.literal_eval(ast.parse(old).body[0].value)
    assert p['id']=='AdaRES-Y88' and p['code_commit']==c['code_commit']=='cb4a5813755d38b3d1593bb803726aabd7d0a79b' and p['gpu']==0 and p['power']==350.
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0]
    root=ROOT/('pipeline/analyses/y88_interrupted_review_'+token);assert not root.exists();root.mkdir();meta=root/'remote_manifest.json'
    value=remote(server,'Q='+repr(dict(folder=job['folder'],pid=job['pid'],job_token=job['token'],program_sha256=ref(closure.artifact(ROOT,job['program']))['sha256']))+'\n'+RECONCILE,meta)
    for row in value['files']:
        path=root/row['name'];assert not path.exists() and path.resolve().is_relative_to(root.resolve());subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',server['user']+'@'+server['host']+':'+row['path'],str(path)],check=True,timeout=90);assert path.stat().st_size==row['bytes'] and ref(path)['sha256']==row['sha256']
    capture=tick.load(root/'adares-y88.json');assert validate_capture(capture,p['methods'][0])
    mutations=[lambda v:v.update(complete=False),lambda v:v.update(completed_passes=1),lambda v:v.update(seed=2000),lambda v:v.update(checkpoint_sha256='0'*64),lambda v:v.update(config_sha256='0'*64),lambda v:v.update(size_cuts_from_training=[1488,3325]),lambda v:v['passes'][0]['rows'].pop()]
    for mutate in mutations:
        bad=copy.deepcopy(capture);mutate(bad)
        try:validate_capture(bad,p['methods'][0])
        except (AssertionError,KeyError,ValueError):pass
        else:raise AssertionError('Accepted malformed completed capture')
    files={r['name']:r for r in value['files']};new=copy.deepcopy(p);new.update(token=token,folder=p['folder'].rsplit('/',1)[0]+'/'+token+'_remaining')
    new['resume_captures']={'AdaRES-Y88':dict(capture_path=files['adares-y88.json']['path'],capture_sha256=files['adares-y88.json']['sha256'],log_path=files['adares-y88.log']['path'],log_sha256=files['adares-y88.log']['sha256'],method=p['methods'][0])}
    assert {k for k in p if p[k]!=new[k]}=={'token','folder'} and set(new)-set(p)=={'resume_captures'}
    header=old.split('\n',1)[1];needle='def main():\n';assert header.count(needle)==1;header=header.replace(needle,REUSE+'\n'+needle)
    needle="captures={};state(dict(status='guarding',completed_methods=[]))";assert header.count(needle)==1;header=header.replace(needle,"captures=reuse_completed(P,folder);state(dict(status='guarding',completed_methods=list(captures)))")
    needle="        for row in P['methods']:\n";assert header.count(needle)==1;header=header.replace(needle,needle+"            if row['id'] in captures:continue\n")
    code='P='+repr(new)+'\n'+header;ast.parse(code,feature_version=(3,8))
    for key in ('methods','freeze','terminal','source_files','adapter_source','adapter_sha256','strip_bootstrap','analysis_source','analysis_sha256','python_paths'):assert new[key]==p[key]
    ast.parse(code,feature_version=(3,8));program=ROOT/('pipeline/checks/y88_remaining_review_'+token+'.py');plan=program.with_suffix('.json');assert not program.exists() and not plan.exists();program.write_text(code,encoding='utf-8',newline='\n')
    disposition=root/'partial_capture_disposition.json';tick.save(disposition,dict(complete=True,id='AdaRES-Y88',token=token,original_job=job,status='resource_guard_failed_between_methods',completed_methods=['AdaRES-Y88'],completed_native_passes=2,each_expressions=9508,each_scenes=141,completed_capture_verified=ref(root/'adares-y88.json'),completed_log=ref(root/'adares-y88.log'),missing_methods=['AdaRES-Y36','AdaRES-W2','AdaRES-B1'],old_worker_and_child_absent=True,missing_baselines_never_started=True,full_four_method_review_complete=False,formal_comparison_not_computed=True,scientific_rejection=False,original_prediction_replayed=False,evidence=[ref(meta),ref(root/'state.json'),ref(root/'worker.log'),job['program'],job['launch'],ref(__file__)]))
    tick.save(plan,dict(complete=True,id='AdaRES-Y88',token=token,program=ref(program),original_job=job,original_preparation=c['frozen_review_execution_binding'],original_launch_binding=c['frozen_review_launch_binding'],partial_capture_disposition=ref(disposition),resume_methods=['AdaRES-Y36','AdaRES-W2','AdaRES-B1'],reuse_completed_methods=['AdaRES-Y88'],completed_capture_identity_coverage_boundary_checks_passed=8,original_full_candidate_capture_verified=True,orchestration_changes_only=['reuse hash-bound completed candidate capture/log byte-for-byte','skip only that completed candidate method','current token/folder'],native_evaluator_adapter_bootstrap_analysis_payload_byte_identical=True,source_checkpoint_config_seed_data_resource_and_original_science_unchanged=True,resource=dict(server='server-1',gpu=0,power_limit_watts=350.),GPU_used=False,checkpoint_loaded=False,launched=False,evidence=[ref(__file__),ref(program),ref(disposition),ref(meta),c['post_training_freeze'],c['terminal_evidence'],c['frozen_review_execution_binding']]))
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y88');assert c['frozen_review_job']==job and not c.get('frozen_review_continuation_plan');c['frozen_review_job'].update(complete=False,failed=True,worker_absent=True,resource_guard_failure=True,completed_methods=['AdaRES-Y88']);c['frozen_review_attempts']=[dict(attempt=1,job=job,intent=c['frozen_review_launch_intent'],disposition=ref(disposition))];c['partial_frozen_review_capture']=ref(disposition);c['frozen_review_continuation_plan']=ref(plan);c['scientific_status']='two_candidate_passes_verified_missing_three_baselines_resource_hold';c['evidence'].extend([ref(plan),ref(disposition),ref(meta),ref(root/'adares-y88.json'),ref(root/'worker.log'),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:lock.close()
    print(json.dumps(dict(plan=ref(plan),partial_capture=ref(disposition),completed_passes_verified=2,remaining_methods=['AdaRES-Y36','AdaRES-W2','AdaRES-B1'],boundary_checks=8,compute_pids=value['compute_pids'],processes=value['processes'],GPU_used=False,launched=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('prepare',));parser.add_argument('--token',required=True);prepare(parser.parse_args().token)
