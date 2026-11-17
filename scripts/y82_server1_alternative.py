"""Prepare/test an inactive Y82 alternative in the verified private server-1 runtime."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from prepare_y69_server1_alternative import ADAPT, adapted_body, idle, observe
import closure

CID = 'AdaRES-Y82'
COMMIT = '5773768ffb87d7fea7432e85fd0da561556aa2fc'
HEAD = '4157937d0c3cf344a44394ce15c95349aa94d50f'
DATA_PROOF = dict(path='pipeline/checks/server1_data_verified_63b27dd8efe04a0bab6dc0291cefdbf0.json',
                  sha256='8c590fa828fbc6c0df840c21ba88b1802cc0d2323eb0a124a57aa328b0c2361f')




def blockers(queue, running):
    result = []
    for task in queue['gpu_queues']['server-1']:
        if task['status'] in ('waiting', 'claiming', 'running'):
            assert str(task['gpu']) == '0'
            result.append(task['id'])
    for gpu, cid in running.get('server-1', {}).items():
        assert str(gpu) == '0'
        result.append(cid)
    return sorted(set(result))

def require_available(queue, running, observation):
    assert not blockers(queue, running), 'Canonical reservation/queue must be released first'
    assert idle(observation), 'Preserve existing/foreign compute and unchanged350W'

def configuration():
    server = monitor.load_servers(str(ROOT))['server-1']; closure.validate_adares_server(ROOT, server)
    assert server['expected_hostname'] == 'dl02-MS-7B73' and server['allowed_gpus'] == [0]
    assert server['python'] == '/data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python'
    assert server['runtime_library_path'] == '/data1/songcl/Project/AdaRES_project/runtime/cuda113lib64'
    assert server['data_root'].rstrip('/') == '/data1/songcl/Project/AdaRES_project/data'
    assert server['max_tasks_per_gpu'] == 1 and len(server['runtime_path_mappings']) == 2
    closure.artifact(ROOT, DATA_PROOF)
    return server

def candidate_plan(work):
    c = next(c for c in work['candidates'] if c['id'] == CID)
    assert c['code_commit'] == COMMIT and not c.get('admission_receipt')
    plan = tick.load(closure.artifact(ROOT, c['native_smoke_plan_server1']))
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(c, plan, frozen) == 7
    assert frozen['runtime_server'] == 'server-1' and frozen['runtime_physical_gpu'] == 0
    assert frozen['runtime_power_limit_watts'] == 350.0
    return c, plan, program, frozen

def probe(token, gpu):
    server = configuration(); work = tick.load(ROOT / 'pipeline/research_work.json'); c, plan, native_program, frozen = candidate_plan(work)
    assert frozen['runtime_path_mappings'] == server['runtime_path_mappings']
    closure.artifact(ROOT, frozen['resource_authorization'])
    if gpu:
        assert c['stage'] == 'validation' and c['required_server'] == 'server-1' and c['required_gpu'] == 0
        assert c['native_smoke_plan'] == c['native_smoke_plan_server1']
        require_available(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)), observe(server))
    program_path = native_program if gpu else closure.artifact(ROOT, plan['CPU_program'])
    name = 'y82_server1_smoke' if gpu else 'y82_server1_CPU_runtime'
    output = ROOT / ('pipeline/analyses/' + name + '_' + token + '.json'); log = output.with_suffix('.log')
    assert not output.exists() and not log.exists()
    remote_output = '/data1/songcl/Project/AdaRES_project/pipeline/probes/' + token + '/' + name + '.json'
    payload = dict(output=remote_output, root=server['remote_code_root'], physical_gpu=0 if gpu else None,
        path_mappings=server['runtime_path_mappings'], token=token, source_sha256=ref(program_path)['sha256'])
    prefix = "import os,sys,pathlib,json,subprocess\nP=" + repr(payload) + "\n"
    prefix += "assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'\n"
    prefix += "assert not pathlib.Path(P['output']).exists()\npathlib.Path(P['output']).parent.mkdir(parents=True,exist_ok=True)\n"
    prefix += "os.environ['PYTHONDONTWRITEBYTECODE']='1'\nos.environ['CUDA_VISIBLE_DEVICES']=" + repr('0' if gpu else '') + "\n"
    prefix += "os.chdir(P['root'])\nsys.path[:0]=" + repr([server['remote_code_root'], server['remote_code_root'] + '/tools',
        server['remote_code_root'] + '/ipdn/lib', server['site_packages'], server['pointnet_egg']]) + '\n'
    if gpu:
        prefix += "import fcntl\ngpu_lease=open('/data1/songcl/Project/AdaRES_project/.foreground_gpu0_diagnostic.lock','a')\nfcntl.flock(gpu_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        prefix += "used,util=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().split(',')]\nassert used<1024 and util<10\nassert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()\n"
    source = prefix + program_path.read_text(encoding='utf-8'); ast.parse(source, feature_version=(3, 8))
    seconds = 600 if gpu else 480
    command = 'LD_LIBRARY_PATH=' + shlex.quote(server['runtime_library_path']) + ' timeout --signal=TERM --kill-after=10s ' + str(seconds) + ' ' + shlex.quote(server['python']) + ' -B -S -'
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    started = datetime.now().astimezone().isoformat()
    with log.open('xb') as stream:
        r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', server['user'] + '@' + server['host'], command], input=source.encode(), stdout=stream, stderr=subprocess.STDOUT, timeout=seconds + 25)
    lease(token, 60); assert r.returncode == 0, 'Preserve preflight/probe failure; no blind replay'
    subprocess.run(['scp', '-q', server['user'] + '@' + server['host'] + ':' + remote_output, str(output)], check=True, timeout=30)
    result = tick.load(output); assert result['complete'] and result['id'] == CID and result['code_commit'] == COMMIT
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    if gpu:
        assert result['steps'] == result['optimizer_steps'] == 2 and result['backward_finite'] and result['intended_path_observed']
        c['native_smoke_result'] = ref(output)
    else:
        assert result['implementation_test_passed'] and result['configured_TRAIN_data_collated'] and not result['CUDA_initialized']
        assert result['native_task_backward'] is False and result['native_smoke_passed'] is False
        assert result['runtime_data_root'].rstrip('/') == server['data_root'].rstrip('/')
        assert result['runtime_pretrain'] == server['data_root'].rstrip('/') + '/sp_unet_backbone.pth'
        assert result['synthetic_CPU_decoder_optimizer_steps'] == 2 and result['train_points'] == 162738 and result['train_expressions'] == 8
        c['CPU_server1_runtime_preflight'] = ref(output)
    evidence = [ref(output), ref(log), ref(program_path), ref(__file__), c['native_smoke_plan_server1']]
    c['evidence'].extend(evidence)
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='native_smoke' if gpu else 'implementation_test', result='passed',
        command='python -B scripts/y82_server1_alternative.py ' + ('run' if gpu else 'cpu') + ' --token ' + token,
        observation='Actual server-1 original native task two-step result; validate full protocol/admission next.' if gpu else
            'Verified private server-1 runtime with CUDA hidden constructed native-YAML full model/released backbone and real mapped original TRAIN data; exact frozen observer callbacks passed two synthetic decoder optimizer steps. No GPU/native sparse task backward/activation/admission or efficacy result.', evidence=evidence)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(result=ref(output), log=ref(log), started_at=started,
        observed_end_local=datetime.now().astimezone().isoformat(), gpu_executed=gpu,
        train_points=result.get('train_points'), train_expressions=result.get('train_expressions'),
        synthetic_decoder_optimizer_steps=result.get('synthetic_CPU_decoder_optimizer_steps'), activated=False if not gpu else True)))
def prepare(token):
    server=configuration();work=tick.load(ROOT/'pipeline/research_work.json')
    c=next(c for c in work['candidates'] if c['id']==CID)
    assert c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' and c['code_commit']==COMMIT
    assert not c.get('admission_receipt') and not c.get('native_smoke_plan_server1')
    assert closure.git(ROOT,'rev-parse','HEAD')==HEAD
    done=monitor.load_experiment(str(ROOT),'AdaRES-Y79')
    assert done['status']=='done_complete' and str(done['gpu'])=='0'
    terminal=tick.load(closure.artifact(ROOT,done['terminal_evidence']))
    assert terminal['completed_epochs']=={p:list(range(1,31)) for p in ('Train','Val')}
    closure.artifact(ROOT,next(x for x in work['candidates'] if x['id']=='AdaRES-Y79')['terminal_archive_binding'])
    qref,rref=ref('pipeline/queue.json'),ref('pipeline/running.json')
    actual=observe(server);require_available(monitor.load_queue(str(ROOT)),monitor.load_running(str(ROOT)),actual)
    original_ref=c['native_smoke_plan_server2_gpu0'];plan=tick.load(closure.artifact(ROOT,original_ref))
    source_path=closure.artifact(ROOT,plan['program']);source=source_path.read_text(encoding='utf-8')
    frozen=ast.literal_eval(ast.parse(source).body[0].value)
    assert validate_frozen(c,plan,frozen)==7
    original=copy.deepcopy(frozen)
    verification=tick.load(closure.artifact(ROOT,c['CPU_native_observer_verification']))
    assert verification['exact_callback_functions_and_classes']==12 and verification['CPU_native_callbacks_actually_executed'] is True
    core_path=ROOT/'scripts/y82_observer_core.py'
    callbacks={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(core_path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
    assert len(callbacks)==12
    for path in (source_path,closure.artifact(ROOT,verification['CPU_program'])):
        found={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in callbacks}
        assert found==callbacks
    base=ROOT/('pipeline/checks/y82_server1_'+token)
    program=base.with_name(base.name+'_native_program.py');ppath=base.with_name(base.name+'_plan.json')
    binding=base.with_name(base.name+'_binding.json');tests=base.with_name(base.name+'_guards.log')
    assert not any(p.exists() for p in (program,ppath,binding,tests))
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','scripts','-p','test_y82_server1_alternative.py','-v'],cwd=ROOT,capture_output=True,timeout=60)
    tests.write_bytes(result.stdout+result.stderr);assert result.returncode==0 and b'Ran 8 tests' in tests.read_bytes()
    frozen.update(runtime_server='server-1',runtime_physical_gpu=0,runtime_power_limit_watts=350.,runtime_path_mappings=copy.deepcopy(server['runtime_path_mappings']),resource_authorization=ref('pipeline/authorizations/server1_gpu0_20261025.json'))
    closure.artifact(ROOT,frozen['resource_authorization'])
    assert all(frozen[k]==original[k] for k in ('variant','code_commit','goal_sha256','train_source_lf_sha256'))
    original_body=source.partition('\n')[2];body=adapted_body(original_body,350.)
    assert body.replace(ADAPT,'').replace("== 'dl02-MS-7B73'","== 'dl01-MS-7C60'")==original_body
    program.write_text('FROZEN='+repr(frozen)+'\n'+body,encoding='utf-8',newline='\n')
    ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    alternate=copy.deepcopy(plan)
    alternate.update(program=ref(program),server='server-1',gpu=0,runtime_power_limit_watts=350.,resource_authorization=frozen['resource_authorization'],runtime_path_mappings=frozen['runtime_path_mappings'],previous_plan=original_ref,
        planned_command='python -B scripts/y82_server1_alternative.py run --token <CURRENT_TOKEN>',
        required_resource='Fresh exclusive idle/unreserved server-1 GPU0, verified private runtime, exact original data and deterministic YAML mappings, unchanged350W; Y79 fixed30 canonically archived. Preserve Y80/Y81 and paused Y57.',
        activation_required=False,activated=True,native_executed=False)
    assert validate_frozen(c,alternate,frozen)==7
    tick.save(ppath,alternate)
    tick.save(binding,dict(complete=True,id=CID,token=token,original_plan=original_ref,alternative_plan=ref(ppath),original_model_and_YAML_payload_identical=True,code_commit_and_goal_unchanged=True,diagnostic_scientific_body_reversible_exact=True,exact_callback_functions=12,CPU_callback_evidence=c['CPU_native_observer_verification'],exact_original_data=DATA_PROOF,actual_server1_observation=actual,guard_tests_passed=8,guard_log=ref(tests),Y79_terminal=done['terminal_evidence'],native_executed=False,hardware_changed=False,production_sync=False,queue_changed=False,evidence=[ref(__file__),ref('scripts/test_y82_server1_alternative.py'),ref(core_path)]))
    c.update(stage='validation',required_server='server-1',required_gpu=0,native_smoke_plan_server1=ref(ppath),native_smoke_plan=ref(ppath),next_action='Execute exact original native YAML/TRAIN two-step task on verified private server-1 runtime and original mapped data now; only actual PASS permits original full frozen evaluation/cost freeze, admission and verified dispatch.')
    for k in ('blocker','blocker_kind','blocked_from_stage','retry_after','unblock_condition','occupation_evidence'):c.pop(k,None)
    c['evidence'].extend([ref(ppath),ref(program),ref(binding),ref(tests),ref(__file__)])
    assert qref==ref('pipeline/queue.json') and rref==ref('pipeline/running.json')
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(binding=ref(binding),active_plan=ref(ppath),tests=8,native_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('prepare','run'));p.add_argument('--token',required=True);a=p.parse_args()
    lease(a.token,800 if a.operation=='run' else 240)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        if a.operation=='prepare':prepare(a.token)
        else:probe(a.token,True)
    finally:lock.close()
