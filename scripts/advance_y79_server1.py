"""Y79 native query-to-language feedback evidence, admission and canonical server1 GPU0 dispatch."""
import argparse
import ast
import copy
from datetime import datetime
import json
import math
import shlex
import subprocess
import sys
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from y79_deployment import Y79
import closure
import experiment_first

SPECS={'AdaRES-Y79':('server-1',0,350.,Y79,'y79_server1_smoke','server1_gpu0_20261025.json')}

def live(server,gpu):
    program="import json,subprocess\nassert subprocess.check_output(['hostname']).decode().strip()=="+repr(server['expected_hostname'])+"\nprint(json.dumps(dict(gpu=subprocess.check_output(['nvidia-smi','-i',"+repr(str(gpu))+",'--query-gpu=utilization.gpu,memory.used,power.limit','--format=csv,noheader,nounits']).decode().strip(),compute=subprocess.check_output(['nvidia-smi','-i',"+repr(str(gpu))+",'--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip())))"
    return json.loads(monitor.ssh(server['host'],server['user'],shlex.quote(server['python'])+' -B -c '+shlex.quote(program),timeout=30))

def idle(v,power):
    util,used,watts=map(float,v['gpu'].split(','))
    assert all(math.isfinite(x) for x in (util,used,watts))
    assert 0<=util<10 and 0<=used<1024 and watts==power and not v['compute'].strip()

def unreserved(queue,running,name,gpu):
    assert not any(t['status'] in ('waiting','claiming','running') and str(t.get('gpu'))==str(gpu) for t in queue['gpu_queues'].get(name,[]))
    assert not running.get(name,{}).get(str(gpu))
    paused=next(t for t in queue['gpu_queues']['server-2'] if t['id']=='AdaRES-Y57')
    assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']

def prepared(cid,work):
    c=next(c for c in work['candidates'] if c['id']==cid)
    name,gpu,power,commit,_,grant=SPECS[cid]
    assert c['code_commit']==commit and c['required_server']==name and c['required_gpu']==gpu
    plan=tick.load(closure.artifact(ROOT,c['native_smoke_plan']));program=closure.artifact(ROOT,plan['program'])
    frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(c,plan,frozen)==7
    assert frozen['runtime_physical_gpu']==gpu and frozen['runtime_power_limit_watts']==power
    closure.artifact(ROOT,frozen['resource_authorization'])
    return c,plan,program,frozen

def validate_result(raw,c,frozen):
    cid=c['id'];name,gpu,power,commit,_,grant=SPECS[cid]
    assert raw['complete'] is True and raw['id']==cid==frozen['variant']['id']
    assert raw['config']==c['config']==frozen['variant']['config']
    assert raw['code_commit']==commit==frozen['code_commit']
    assert raw['goal_sha256']==frozen['goal_sha256'] and raw['native_YAML_sha256']==frozen['variant']['config_data']['sha256']
    assert raw['steps']==raw['optimizer_steps']==2
    assert len(raw['losses'])==2 and all(math.isfinite(x) for x in raw['losses'])
    assert raw['native_entry']=='tools/train.py::train' and raw['split']=='train'
    assert all(raw[k] is True for k in ('native_data_loaded','backward_finite','intended_path_observed','native_modes_verified'))
    assert raw['validation_used'] is False and raw['model_weights_saved'] is False
    assert raw['shared_head_calls']==raw['query_norm_consumption_calls']==14
    assert raw['feedback_calls']==raw['qla_consumption_calls']==12 and raw['next_lla_consumption_calls']==10
    assert raw['static_mask_features_preserved'] is True and raw['original_initial_prediction_preserved'] is True
    assert len(raw['gradient_checks'])==len(raw['batches'])==2
    from test_y79_native_smoke_plan import N
    N['check_response_rows'](raw['feedback_paths'])
    for row in raw['gradient_checks']:
        assert row['all_finite'] is True and row['tensors']>0 and row['indicator_tensors']==10 and row['indicator_calls']==7
        assert row['feedback_calls']==row['qla_consumption_calls']==6 and row['next_lla_consumption_calls']==5 and row['query_norm_consumption_calls']==7
        assert row['both_input_gradients_finite'] is True
        assert all(math.isfinite(row[k]) and row[k]>0 for k in ('feedback_weight_gradient_max','parameter_update_max'))
    assert 0<raw['peak_reserved_bytes']<24*2**30
    assert all(len(b['scan_ids'])==2 and b['expressions']>0 and b['points']>0 for b in raw['batches'])
    assert len({scene for b in raw['batches'] for scene in b['scan_ids']})==4
    assert raw['data_source']['dataset_class']=='ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'
    expected='/data1/songcl/Project/AdaRES_project/data' if name=='server-1' else '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data'
    assert raw['data_source']['config_data_root'].rstrip('/')==expected

def admit(token,cid):
    name,gpu,power,commit,label,grant=SPECS[cid];server=monitor.load_servers(str(ROOT))[name];closure.validate_adares_server(ROOT,server)
    work=tick.load(ROOT/'pipeline/research_work.json');c,plan,program,frozen=prepared(cid,work)
    assert c['stage']=='validation' and not c.get('admission_receipt')
    assert closure.git(ROOT,'rev-parse','HEAD')==commit
    rawpath=closure.artifact(ROOT,c['native_smoke_result']);raw=tick.load(rawpath);validate_result(raw,c,frozen)
    queue=monitor.load_queue(str(ROOT));unreserved(queue,monitor.load_running(str(ROOT)),name,gpu)
    assert monitor.load_experiment(str(ROOT),cid) is None
    current=live(server,gpu);idle(current,power)
    cfg=yaml.safe_load((ROOT/'AdaRES'/c['config']).read_bytes())
    assert cfg['train']['seed']==1999 and cfg['train']['epochs']==30 and cfg['dataloader']['train']['batch_size']==2
    assert cfg['model']['sampling_module']['num_proposal']==128
    tests=ROOT/('pipeline/checks/'+cid.lower()+'_admission_guards_'+token+'.log');assert not tests.exists()
    test=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','scripts','-p','test_y79_admission.py','-v'],cwd=ROOT,capture_output=True,timeout=60)
    tests.write_bytes(test.stdout+test.stderr);assert test.returncode==0
    parentref=next(c for c in work['candidates'] if c['id']=='AdaRES-Y64')['post_training_freeze'];parent=tick.load(closure.artifact(ROOT,parentref))
    prefix='pipeline/checks/'+cid+'_'+token;smokepath=ROOT/(prefix+'_native_smoke.json');freezepath=ROOT/(prefix+'_post_training_freeze.json');receiptpath=ROOT/('pipeline/admissions/'+cid+'.json')
    assert not any(p.exists() for p in (smokepath,freezepath,receiptpath))
    command='python -B scripts/y79_server1_alternative.py run --token '+token
    smoke=copy.deepcopy(raw);smoke.update(command=command,server=name,physical_gpu=gpu,admission_resource_check=current,
        evidence=[ref(rawpath),ref(rawpath.with_suffix('.log')),ref(program),c['native_smoke_plan']]);tick.save(smokepath,smoke)
    freeze=copy.deepcopy(parent);freeze.update(id=cid,frozen_at=datetime.now().astimezone().isoformat(),goal=ref('pipeline/goal.json'),
        code_commit=commit,parent_protocol=parentref,resource_condition=dict(server=name,gpu=gpu,power_limit_watts=power,
            note='Existing power preserved; paired evaluation/cost on same device; two native batches are not a full-dataset memory bound.'),
        resource_authorization=ref('pipeline/authorizations/'+grant),scientific_efficacy='Unknown until original frozen full evaluation.')
    freeze['evaluation']['methods'][0]=cid;restored=copy.deepcopy(freeze['evaluation']);restored['methods'][0]='AdaRES-Y64'
    assert restored==parent['evaluation'] and freeze['training']==parent['training'];tick.save(freezepath,freeze)
    receipt=dict(profile='experiment_first',id=cid,candidate_id=cid,config=c['config'],code_commit=commit,goal_sha256=raw['goal_sha256'],
        authorization=experiment_first.effective(ROOT)['authorization'],native_smoke=ref(smokepath),post_training_plan=ref(freezepath),
        resource_authorization=ref('pipeline/authorizations/'+grant),admitted_at=datetime.now().astimezone().isoformat(),
        gates={g:dict(passed=True,evidence=[ref(smokepath),ref(freezepath),ref(program)]) for g in experiment_first.GATES})
    experiment_first.validate_smoke(ROOT,receipt)
    decision=monitor.log_decision(str(ROOT),'next_batch',cid+'是否通过真实原生检查并可独占训练？',
        '原生YAML/精确映射TRAIN两步、六原query/language-FFN/padding身份、未detach与六qla和五next-lla消费者、七共享confidence/query_norm、首零次响应/有限新梯度及参数更新通过，保持原完整后评协议并入队。',
        '固定30/seed1999、独立配置与现有授权功耗不变；效益待原冻结复评，保留Y57暂停和全部存活任务。',ref(smokepath)['path'])
    receipt['decision']=ref(decision);tick.save(receiptpath,receipt)
    task=dict(id=cid,label=cid+' fixed30',config=c['config'],status='waiting',gpu=gpu,server=name,depends_on=[],code_commit=commit,
        admission_receipt=ref(receiptpath)['path'],scientific_candidate_id=cid,patience=30,min_epochs=30,fixed_epochs=30)
    closure.admission(ROOT,task)
    c.update(stage='admission',admission_receipt=task['admission_receipt'],native_smoke=ref(smokepath),post_training_freeze=ref(freezepath),next_action='Canonical gated dispatch; confirm increasing native steps. Preserve other jobs and Y57 pause.')
    c['evidence'].extend([ref(smokepath),ref(freezepath),ref(receiptpath),ref(decision),ref(tests),ref(__file__)])
    work['tick_attempt']=dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],candidate_id=cid,
        kind='native_smoke',result='passed',command=command,observation='Actual native two-step paths/backward/optimizer passed; queued with original complete frozen evaluation.',evidence=[ref(smokepath),ref(freezepath)])
    tick.save(ROOT/'pipeline/research_work.json',work);queue['gpu_queues'][name].append(task);monitor.save_queue(str(ROOT),queue)
    print(json.dumps(dict(admitted=cid,receipt=ref(receiptpath),losses=raw['losses'])))

def dispatch(token,cid):
    name,gpu,power,commit,_,grant=SPECS[cid];server=monitor.load_servers(str(ROOT))[name];queue=monitor.load_queue(str(ROOT))
    tasks=queue['gpu_queues'][name];task=next(t for t in tasks if t['id']==cid);assert task['status']=='waiting'
    closure.admission(ROOT,task);assert not monitor.load_running(str(ROOT)).get(name,{}).get(str(gpu))
    assert not any(t['id']!=cid and t['status'] in ('waiting','claiming','running') and str(t.get('gpu'))==str(gpu) for t in tasks)
    others={s:copy.deepcopy([t for t in ts if t['id']!=cid]) for s,ts in queue['gpu_queues'].items()}
    current=live(server,gpu);idle(current,power)
    active={str(t['gpu']) for t in tasks if t['status'] in ('running','claiming')};events=[]
    monitor._process_queue(queue,name,server['host'],server['user'],str(ROOT),active,{str(gpu)},events,
        max_tasks_per_gpu=1,allow_external_overlay=False,remote_code_root=server['remote_code_root'],current_tasks=len(active),max_tasks_total=len(server['allowed_gpus']))
    queue=monitor.load_queue(str(ROOT));task=next(t for t in queue['gpu_queues'][name] if t['id']==cid)
    assert {s:[t for t in ts if t['id']!=cid] for s,ts in queue['gpu_queues'].items()}==others
    out=ROOT/('pipeline/checks/'+cid.lower()+'_dispatch_'+token+'.json');assert not out.exists()
    tick.save(out,dict(complete=True,token=token,id=cid,before=current,task=task,events=events,other_queue_entries_preserved=True,helper=ref(__file__)))
    work=tick.load(ROOT/'pipeline/research_work.json');c=next(c for c in work['candidates'] if c['id']==cid)
    c.update(stage='training' if task['status']=='running' and task.get('launch_verified') else 'deployment',next_action='Observe exact native phase/step increase; reconcile uncertain launch without restart.')
    c['evidence'].append(ref(out));tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(evidence=ref(out),status=task['status'],pid=task.get('pid'),launch_verified=task.get('launch_verified'),events=events)))


def verify(token):
    work=tick.load(ROOT/'pipeline/research_work.json');c,plan,program,frozen=prepared('AdaRES-Y79',work)
    assert c['stage']=='validation' and c.get('native_smoke_result')
    path=closure.artifact(ROOT,c['native_smoke_result']);raw=tick.load(path);validate_result(raw,c,frozen)
    print(json.dumps(dict(result=ref(path),losses=raw['losses'],peak_reserved_bytes=raw['peak_reserved_bytes'],actual_native_verified=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('verify','admit','dispatch'));p.add_argument('--token',required=True);a=p.parse_args()
    lease(a.token,300);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        if a.operation=='verify':verify(a.token)
        else:dict(admit=admit,dispatch=dispatch)[a.operation](a.token,'AdaRES-Y79')
    finally:lock.close()
