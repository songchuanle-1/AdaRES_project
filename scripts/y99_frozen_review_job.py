"""Advance the already archived Y99 original server1 review using unchanged science and explicit additive source binding."""
import argparse,ast,copy,hashlib,inspect,json,shlex,subprocess
from datetime import datetime
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y102_deployment import git,Y101,Y102,adapt
from y101_frozen_review_job import WORKER,STRIP_BOOTSTRAP,remote
from analyze_y23_frozen_terminal import comparison as corrected_comparison
import launch_y101_frozen_review_job as old_observer
import closure,yaml

CID='AdaRES-Y99';ORIGINAL_COMMIT='53f5b028bdf24e8b41dd934931d1fedc99843d8d';OLD_PID=2980836

def worker_source():
    source=WORKER.replace('AdaRES-Y101',CID).replace('adares_y101_query_ffn_skip_res','adares_y99_spatial_mask_skip_res').replace('20261112_200238','20261111_180638').replace('2263581',str(OLD_PID))
    needle="(root/'.codex_source_commit').read_text().strip()==P['code_commit']";assert source.count(needle)==1;source=source.replace(needle,"(root/'.codex_source_commit').read_text().strip()==P['runtime_marker']")
    needle="ns['comparison'](captures['AdaRES-Y99'],captures[cid],seed=1999,draws=10000)";assert source.count(needle)==1;source=source.replace(needle,"ns['comparison'](captures['AdaRES-Y99'],captures[cid])")
    needle="assert p['id']=='AdaRES-Y99'";assert source.count(needle)==1;source=source.replace(needle,"assert p['code_commit']==p['freeze']['code_commit']=='"+ORIGINAL_COMMIT+"' and p['runtime_marker']=='"+Y101+"'\n    "+needle)
    ast.parse(source,feature_version=(3,8));return source

def prepare(token):
    lease(token,600);assert git('rev-parse','HEAD').decode().strip()==Y102
    w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']==CID);done=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');assert done['stage']=='complete' and tick.load(closure.artifact(ROOT,done['complete_scientific_review']))['full_cost_and_postaudit_complete']
    assert c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' and c['required_server']=='server-1' and c['required_gpu']==0 and not c.get('frozen_review_job')
    terminal=tick.load(closure.artifact(ROOT,c['terminal_evidence']));archive=tick.load(closure.artifact(ROOT,c['terminal_archive_binding']));freeze=tick.load(closure.artifact(ROOT,c['post_training_freeze']));assert monitor.load_experiment(str(ROOT),CID)['status']=='done_complete'
    assert terminal['checkpoints'][0]['epoch']==30 and terminal['checkpoints'][1]['epoch']==23 and archive['complete'];assert freeze['code_commit']==c['code_commit']==ORIGINAL_COMMIT and freeze['resource_condition']['server']=='server-1' and freeze['resource_condition']['gpu']==0 and freeze['resource_condition']['power_limit_watts']==350.
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0]
    old_program=closure.artifact(ROOT,done['frozen_review_job']['program']);old=ast.literal_eval(ast.parse(old_program.read_text(encoding='utf-8')).body[0].value)
    native_config=server['remote_code_root']+'/'+c['config'];expected=adapt(yaml.safe_load(git('show',ORIGINAL_COMMIT+':AdaRES/'+c['config'])),server)
    paths=[p for p in git('ls-tree','-r','--name-only',ORIGINAL_COMMIT,'AdaRES/ipdn/model_paper5/adares_y99_spatial_mask_skip').decode().splitlines() if p.endswith('.py')];assert len(paths)==6
    sources=[r for r in old['source_files'] if 'adares_y101_query_ffn_skip/' not in r['path']];assert len(sources)==20
    sources.extend(dict(path=p.removeprefix('AdaRES/'),sha256=hashlib.sha256(git('show',ORIGINAL_COMMIT+':'+p)).hexdigest(),LF=False) for p in paths);assert len(sources)==26
    runtime=tick.load(closure.artifact(ROOT,done['frozen_review_runtime_binding']));first_program=closure.artifact(ROOT,runtime['results'][0]).with_suffix('.py');assert first_program.is_file();text=first_program.read_text(encoding='utf-8');first=ast.literal_eval(ast.parse(text).body[0].value);body=text.split('\n',1)[1]
    first.update(id=CID,config=native_config,expected_config=expected,config_sha256=None,expected_module='ipdn.model_paper5.adares_y99_spatial_mask_skip.ipdn')
    check="import hashlib,pathlib,subprocess\nassert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'\nassert (pathlib.Path(P['code'])/'.codex_source_commit').read_text().strip()=="+repr(Y101)+"\nassert not pathlib.Path('/proc/2980836').exists()\nfor row in SOURCES:\n raw=(pathlib.Path(P['code'])/row['path']).read_bytes();raw=raw.replace(b'\\r\\n',b'\\n') if row['LF'] else raw\n assert hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']\n"
    program='P='+repr(first)+'\nSOURCES='+repr(sources)+'\n'+check+body;ast.parse(program,feature_version=(3,8))
    base='pipeline/checks/y99_review_binding_'+token;path=ROOT/(base+'_import.py');out=ROOT/(base+'_import.json');assert not path.exists() and not out.exists();path.write_text(program,encoding='utf-8',newline='\n')
    prefix='PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH='+shlex.quote(server['runtime_library_path'])+' '+shlex.quote(server['python'])+' -B -S -'
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',server['user']+'@'+server['host'],prefix],input=program.encode(),capture_output=True,timeout=150);out.with_suffix('.log').write_bytes(r.stdout+r.stderr);assert r.returncode==0,'Preserve binding failure, do not launch or alter model/environment blindly';value=json.loads(r.stdout.decode().splitlines()[-1]);tick.save(out,value);assert value['complete'] and not value['CUDA_initialized'] and not value['checkpoint_loaded']
    p=copy.deepcopy(old);p.update(id=CID,token=token,old_pid=OLD_PID,code_commit=ORIGINAL_COMMIT,runtime_marker=Y101,candidate_config=c['config'],freeze=freeze,terminal=terminal,source_files=sources,folder='/data1/songcl/Project/AdaRES_project/pipeline/review_jobs/'+CID+'/'+token)
    p['methods'][0]=dict(id=CID,config=native_config,config_sha256=value['config_sha256'],checkpoint=terminal['checkpoints'][1]['path'],checkpoint_sha256=terminal['checkpoints'][1]['sha256'])
    analysis=(ROOT/'scripts/analyze_frozen_full.py').read_text(encoding='utf-8')+'\nimport statistics\nhistorical_comparison=comparison\n'+inspect.getsource(corrected_comparison)+'\n';p['analysis_source']=analysis;p['analysis_sha256']=hashlib.sha256(analysis.encode()).hexdigest()
    source=worker_source();ns={'__name__':'y99_guard_tests'};exec(compile(source,'y99_fixed_review','exec'),ns);assert ns['validate'](p)
    mutations=[('GPU1',lambda q:q.update(gpu=1)),('wrong_server',lambda q:q.update(server='server-2')),('wrong_power',lambda q:q.update(power=280.)),('live_original',lambda q:q['terminal'].update(process_absent=False)),('wrong_runtime_marker',lambda q:q.update(runtime_marker=ORIGINAL_COMMIT)),('wrong_candidate_commit',lambda q:q.update(code_commit=Y101)),('incomplete_last',lambda q:q['terminal']['checkpoints'][0].update(epoch=29)),('nonfinite_best',lambda q:q['terminal']['checkpoints'][1].update(model_finite=False)),('wrong_scheduler',lambda q:q['terminal']['checkpoints'][1].update(scheduler_last_epoch=22)),('missing_val_epoch',lambda q:q['terminal']['completed_epochs']['Val'].pop())]
    for name,fn in mutations:
        bad=copy.deepcopy(p);fn(bad)
        try:ns['validate'](bad)
        except (AssertionError,KeyError,ValueError):pass
        else:raise AssertionError('Accepted '+name)
    execution=ROOT/(base+'_execution.py');assert not execution.exists();execution.write_text('P='+repr(p)+'\n'+source,encoding='utf-8',newline='\n')
    manifest=ROOT/(base+'.json');assert not manifest.exists();tick.save(manifest,dict(complete=True,id=CID,token=token,original_candidate_commit=ORIGINAL_COMMIT,current_runtime_marker=Y101,source_files_verified=26,own_model_files=6,CPU_identity_and_launch_guards_passed=11,private_runtime_config_and_class_import_passed=True,original_native_scientific_tests_not_replayed=True,checkpoint_loaded=False,GPU_used=False,program=ref(execution),own_CPU_import=ref(out),baseline_config_weight_bindings_reused_byte_identically=True,corrected_original_repeat_noise_analysis_reused=True,original_resource=freeze['resource_condition'],native_evaluation_not_started=True,evidence=[ref(__file__),ref(path),ref(out),ref(out.with_suffix('.log')),ref(execution),c['post_training_freeze'],c['terminal_evidence'],c['terminal_archive_binding'],done['complete_scientific_review'],done['frozen_review_resource_adapter'],done['frozen_review_bootstrap_test'],ref('scripts/analyze_frozen_full.py'),ref('scripts/analyze_y23_frozen_terminal.py')]))
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']==CID);assert c['stage']=='blocked' and not c.get('frozen_review_job');c['frozen_review_execution_binding']=ref(manifest);c['evidence'].extend([ref(manifest),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    print(json.dumps(dict(binding=ref(manifest),CPU_import_and_11_guards_passed=True,GPU_used=False)))

def launch(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']==CID);done=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');assert done['stage']=='complete' and c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' and not c.get('frozen_review_job') and not c.get('frozen_review_launch_intent')
    binding=tick.load(closure.artifact(ROOT,c['frozen_review_execution_binding']));assert binding['CPU_identity_and_launch_guards_passed']==11 and binding['private_runtime_config_and_class_import_passed'];program=closure.artifact(ROOT,binding['program']);text=program.read_text(encoding='utf-8');p=ast.literal_eval(ast.parse(text).body[0].value)
    q=monitor.load_queue(str(ROOT));assert not monitor.load_running(str(ROOT)).get('server-1',{}).get('0') and not any(r['status'] in ('running','claiming') and str(r.get('gpu'))=='0' for r in q['gpu_queues']['server-1'])
    s=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,s);assert s['allowed_gpus']==[0]
    intent=ROOT/('pipeline/checks/y99_review_launch_intent_'+token+'.json');assert not intent.exists();tick.save(intent,dict(complete=True,id=CID,token=token,status='intent_before_remote_creation',program=ref(program),binding=c['frozen_review_execution_binding'],remote_folder=p['folder'],original_resource=p['freeze']['resource_condition']))
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']==CID);c['frozen_review_launch_intent']=ref(intent);c['evidence'].append(ref(intent));tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    body=r'''import hashlib,json,os,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
assert not pathlib.Path('/proc/2980836').exists()
assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
assert float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip())==350.
folder=pathlib.Path(Q['folder']);assert str(folder.resolve()).startswith('/data1/songcl/Project/AdaRES_project/pipeline/review_jobs/AdaRES-Y99/') and not folder.exists();folder.mkdir(parents=True)
raw=Q['program'].encode();assert hashlib.sha256(raw).hexdigest()==Q['sha'];path=folder/'worker.py';path.write_bytes(raw)
(folder/'launch_intent.json').write_text(json.dumps(dict(token=Q['token'],program_sha256=Q['sha'],status='launching')))
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=Q['library'],PYTHONPATH=os.pathsep.join(Q['paths']))
with (folder/'worker.log').open('xb') as stream:child=subprocess.Popen([Q['python'],'-B','-S',str(path)],cwd=Q['root'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
value=dict(complete=True,id='AdaRES-Y99',token=Q['token'],pid=child.pid,folder=str(folder),program_sha256=Q['sha'],process_created=True,native_progress_confirmed=False);(folder/'launch.json').write_text(json.dumps(value));print(json.dumps(value))
'''
    out=ROOT/('pipeline/checks/y99_review_launch_'+token+'.json');v=remote(s,'Q='+repr(dict(folder=p['folder'],token=token,program=text,sha=ref(program)['sha256'],python=s['python'],library=s['runtime_library_path'],paths=p['python_paths'],root=s['remote_code_root']))+'\n'+body,out)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']==CID);assert c['stage']=='blocked' and not c.get('frozen_review_job');c.update(stage='validation',scientific_status='fixed30_complete_frozen_review_pending',frozen_review_job=dict(folder=p['folder'],token=token,pid=v['pid'],launch=ref(out),program=ref(program),native_progress_confirmed=False),next_action='Observe exact original-resource Y99 four-method full review job; never retry/duplicate or restart original training. Retain corrected original paired/noise analysis and pending same-device cost/postaudit. Y101 full scientific result remains complete_inconclusive; all GPU1/Y57 holds unchanged.')
        for key in ('blocker','blocker_kind','blocked_from_stage','retry_after','unblock_condition','occupation_evidence'):c.pop(key,None)
        c['evidence'].extend([ref(out),ref(out.with_suffix('.log')),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    print(json.dumps(dict(launch=ref(out),pid=v['pid'],native_progress_confirmed=False,new_training=False)))

def observe(token,label):
    source=inspect.getsource(old_observer.observe);updated=source.replace('AdaRES-Y101',CID).replace('y101_review_progress_','y99_review_progress_').replace('/proc/2263581','/proc/2980836').replace('time.sleep(5)','time.sleep(30)')
    assert updated.replace(CID,'AdaRES-Y101').replace('y99_review_progress_','y101_review_progress_').replace('/proc/2980836','/proc/2263581').replace('time.sleep(30)','time.sleep(5)')==source
    ns=dict(vars(old_observer));ns['__file__']=__file__;exec(compile(ast.parse(updated),__file__,'exec'),ns);ns['observe'](token,label)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('prepare','launch','observe'));parser.add_argument('--token',required=True);parser.add_argument('--label',default='start');a=parser.parse_args()
    if a.operation=='prepare':prepare(a.token)
    elif a.operation=='launch':launch(a.token)
    else:observe(a.token,a.label)
