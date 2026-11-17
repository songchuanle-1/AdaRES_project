"""Prepare an inactive Y84 server-1 plan while preserving the live Y82 run."""
import argparse
import ast
import copy
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from prepare_y69_server1_alternative import ADAPT, adapted_body, idle, observe
from y82_server1_alternative import configuration, blockers, require_available, DATA_PROOF
import closure

CID = 'AdaRES-Y84'
COMMIT = '7bef2cd313b870583c970c56580be5a78097234d'
HEAD = '2818322e6049b2541f1590a4fa9ad6854b3c598d'


def checked_plan(work):
    c = next(c for c in work['candidates'] if c['id'] == CID)
    assert c['code_commit'] == COMMIT and not c.get('admission_receipt')
    plan = tick.load(closure.artifact(ROOT, c['native_smoke_plan_server1']))
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(c, plan, frozen) == 7
    assert frozen['runtime_server'] == 'server-1' and frozen['runtime_physical_gpu'] == 0
    assert frozen['runtime_power_limit_watts'] == 350.0
    return c, plan, program, frozen


def prepare(token):
    server = configuration()
    work = tick.load(ROOT / 'pipeline/research_work.json')
    c = next(c for c in work['candidates'] if c['id'] == CID)
    assert c['stage'] == 'blocked' and c['blocker_kind'] == 'exclusive_gpu_slot'
    assert c['code_commit'] == COMMIT and not c.get('admission_receipt') and not c.get('native_smoke_plan_server1')
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == HEAD
    queue = monitor.load_queue(str(ROOT)); running = monitor.load_running(str(ROOT))
    qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
    live = next(t for t in queue['gpu_queues']['server-1'] if t['id'] == 'AdaRES-Y82')
    assert live['status'] == 'running' and live['launch_verified'] and live['pid'] == 1121817
    assert running['server-1']['0'] == live['id']
    closure.admission(ROOT, live)
    actual = observe(server)
    idle_now = idle(actual)
    original_ref = c['native_smoke_plan_server2_gpu0']
    plan = tick.load(closure.artifact(ROOT, original_ref))
    source_path = closure.artifact(ROOT, plan['program'])
    source = source_path.read_text(encoding='utf-8')
    frozen = ast.literal_eval(ast.parse(source).body[0].value)
    assert validate_frozen(c, plan, frozen) == 7
    original = copy.deepcopy(frozen)
    verification = tick.load(closure.artifact(ROOT, c['CPU_native_observer_verification']))
    assert verification['exact_callback_functions_and_classes'] == 6
    assert verification['CPU_native_callbacks_actually_executed'] is True
    core_path = ROOT / 'scripts/y84_observer_core.py'
    callbacks = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(core_path.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef)}
    assert len(callbacks) == 6
    for path in (source_path, closure.artifact(ROOT, verification['CPU_program'])):
        found = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name in callbacks}
        assert found == callbacks
    base = ROOT / ('pipeline/checks/y84_server1_alternative_' + token)
    program = base.with_name(base.name + '_program.py')
    ppath = base.with_name(base.name + '_plan.json')
    binding = base.with_name(base.name + '_binding.json')
    tests = base.with_name(base.name + '_guards.log')
    assert not any(p.exists() for p in (program, ppath, binding, tests))
    result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_y84_server1_alternative.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    tests.write_bytes(result.stdout + result.stderr)
    assert result.returncode == 0 and b'Ran 8 tests' in tests.read_bytes()
    frozen.update(runtime_server='server-1', runtime_physical_gpu=0, runtime_power_limit_watts=350.,
        runtime_path_mappings=copy.deepcopy(server['runtime_path_mappings']),
        resource_authorization=ref('pipeline/authorizations/server1_gpu0_20261025.json'))
    closure.artifact(ROOT, frozen['resource_authorization'])
    assert all(frozen[k] == original[k] for k in ('variant', 'code_commit', 'goal_sha256', 'train_source_lf_sha256'))
    original_body = source.partition('\n')[2]
    body = adapted_body(original_body, 350.)
    assert body.replace(ADAPT, '').replace("== 'dl02-MS-7B73'", "== 'dl01-MS-7C60'") == original_body
    program.write_text('FROZEN=' + repr(frozen) + '\n' + body, encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    found = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(body).body if isinstance(n, ast.FunctionDef) and n.name in callbacks}
    assert found == callbacks
    alternate = copy.deepcopy(plan)
    alternate.update(program=ref(program), server='server-1', gpu=0, runtime_power_limit_watts=350.,
        resource_authorization=frozen['resource_authorization'], runtime_path_mappings=frozen['runtime_path_mappings'],
        previous_plan=original_ref, planned_command='python -B scripts/y84_server1_alternative.py run --token <CURRENT_TOKEN>',
        required_resource='Only after exact Y82 fixed30 canonical archival and fresh exclusive idle/unreserved server-1 GPU0; original private runtime/data, configured deterministic YAML mappings and existing350W. Preserve Y83 and GPU1/Y57 user pauses.',
        activation_required=True, activated=False, native_executed=False)
    assert validate_frozen(c, alternate, frozen) == 7
    tick.save(ppath, alternate)
    tick.save(binding, dict(complete=True, id=CID, token=token, observed_at=datetime.now().astimezone().isoformat(),
        original_plan=original_ref, inactive_server1_plan=ref(ppath), original_model_and_YAML_payload_identical=True,
        code_commit_and_goal_unchanged=True, diagnostic_scientific_body_reversible_exact=True,
        exact_callback_functions=6, CPU_callbacks_previously_executed=True, CPU_callback_evidence=c['CPU_native_observer_verification'],
        exact_original_data=DATA_PROOF, actual_server1_observation=actual, physical_idle=idle_now,
        canonical_reservation=live['id'], activated=False, guard_tests_passed=8, guard_log=ref(tests),
        native_executed=False, hardware_changed=False, production_sync=False, queue_changed=False,
        evidence=[ref(__file__), ref('scripts/test_y84_server1_alternative.py'), ref(core_path),
            ref('scripts/y82_server1_alternative.py'), ref('scripts/prepare_y69_server1_alternative.py')]))
    c['native_smoke_plan_server1'] = ref(ppath)
    c['server1_alternative_preparation'] = ref(binding)
    c['evidence'].extend([ref(program), ref(ppath), ref(binding), ref(tests), ref(__file__), DATA_PROOF])
    c['next_action'] = ('Y84 precedes Y85. Preserve tested CPU/source/native plans; no replay. '
        'Use the exact server-2 GPU0/280W native plan when exclusively idle. If server-1 GPU0 frees first, '
        'canonically verify/archive original Y82 fixed30, activate the stored server-1 alternative with fresh '
        'idle/unreserved proof, then run its original native YAML/TRAIN two-step task. Only actual PASS permits '
        'complete frozen evaluation/cost, receipt, safe source selection/sync, admission and verified launch. '
        'GPU1 remains paused; never execute its retained plan.')
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B scripts/y84_server1_alternative.py prepare --token ' + token,
        observation='Inactive resource-only server-1 alternative: eight mapping/host/power/idle/reservation guards passed; identical model/YAML/code/goal and six previously executed exact callbacks preserved. Y82 reservation remains. No GPU task, activation, admission or efficacy.',
        evidence=[ref(binding), ref(tests)])
    assert all(e in c['evidence'] for e in work['tick_attempt']['evidence'])
    assert qref == ref('pipeline/queue.json') and rref == ref('pipeline/running.json')
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(binding=ref(binding), inactive_plan=ref(ppath), tests=8, activated=False, native_executed=False)))


def activate(token):
    server = configuration(); work = tick.load(ROOT/'pipeline/research_work.json')
    c, plan, program, frozen = checked_plan(work)
    assert c['stage'] == 'blocked' and c['blocker_kind'] == 'exclusive_gpu_slot'
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == HEAD
    assert frozen['runtime_path_mappings'] == server['runtime_path_mappings']
    closure.artifact(ROOT, frozen['resource_authorization'])
    done = monitor.load_experiment(str(ROOT), 'AdaRES-Y82')
    assert done['status'] == 'done_complete' and str(done['gpu']) == '0'
    terminal = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
    assert terminal['completed_epochs'] == {p: list(range(1, 31)) for p in ('Train', 'Val')}
    assert terminal['checkpoints'][0]['epoch'] == 30
    require_available(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)), observe(server))
    out = ROOT/('pipeline/checks/y84_server1_activation_' + token + '.json'); assert not out.exists()
    tick.save(out, dict(complete=True, token=token, id=CID, activated_at=datetime.now().astimezone().isoformat(),
        active_plan=c['native_smoke_plan_server1'], original_Y82_terminal=done['terminal_evidence'],
        idle_and_no_canonical_reservation=True, native_executed=False, GPU1_use_authorized=False))
    c.update(stage='validation', required_server='server-1', required_gpu=0, native_smoke_plan=c['native_smoke_plan_server1'])
    for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition', 'occupation_evidence'):
        c.pop(key, None)
    c['evidence'].append(ref(out)); c['next_action'] = plan['planned_command'] + '; actual PASS required before admission.'
    tick.save(ROOT/'pipeline/research_work.json', work)
    print(json.dumps(dict(activation=ref(out), native_executed=False)))


def run(token):
    server = configuration(); work = tick.load(ROOT/'pipeline/research_work.json')
    c, plan, program, frozen = checked_plan(work)
    assert c['stage'] == 'validation' and c['required_server'] == 'server-1' and c['required_gpu'] == 0
    assert c['native_smoke_plan'] == c['native_smoke_plan_server1']
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == HEAD
    assert frozen['runtime_path_mappings'] == server['runtime_path_mappings']
    closure.artifact(ROOT, frozen['resource_authorization'])
    require_available(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)), observe(server))
    out = ROOT/('pipeline/analyses/y84_server1_smoke_' + token + '.json'); log = out.with_suffix('.log')
    assert not out.exists() and not log.exists()
    remote_output = '/data1/songcl/Project/AdaRES_project/pipeline/probes/' + token + '/y84_server1_smoke.json'
    payload = dict(output=remote_output, root=server['remote_code_root'], physical_gpu=0,
        path_mappings=server['runtime_path_mappings'], token=token, source_sha256=ref(program)['sha256'])
    prefix = "import os,sys,pathlib,json,subprocess,fcntl\nP=" + repr(payload) + "\n"
    prefix += "assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'\n"
    prefix += "assert not pathlib.Path(P['output']).exists()\npathlib.Path(P['output']).parent.mkdir(parents=True,exist_ok=True)\n"
    prefix += "os.environ['PYTHONDONTWRITEBYTECODE']='1'\nos.environ['CUDA_VISIBLE_DEVICES']='0'\n"
    prefix += "os.chdir(P['root'])\nsys.path[:0]=" + repr([server['remote_code_root'], server['remote_code_root']+'/tools',
        server['remote_code_root']+'/ipdn/lib', server['site_packages'], server['pointnet_egg']]) + '\n'
    prefix += "gpu_lease=open('/data1/songcl/Project/AdaRES_project/.foreground_gpu0_diagnostic.lock','a')\nfcntl.flock(gpu_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
    prefix += "used,util=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().split(',')]\nassert used<1024 and util<10\nassert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()\n"
    source = prefix + program.read_text(encoding='utf-8'); ast.parse(source, feature_version=(3, 8))
    command = 'LD_LIBRARY_PATH=' + shlex.quote(server['runtime_library_path']) + ' timeout --signal=TERM --kill-after=10s 600 ' + shlex.quote(server['python']) + ' -B -S -'
    qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
    with log.open('xb') as stream:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            server['user']+'@'+server['host'], command], input=source.encode(), stdout=stream, stderr=subprocess.STDOUT, timeout=625)
    lease(token, 60); assert result.returncode == 0, 'Preserve failed probe; never blindly replay'
    subprocess.run(['scp', '-q', server['user']+'@'+server['host']+':'+remote_output, str(out)], check=True, timeout=30)
    value = tick.load(out)
    assert value['complete'] and value['id'] == CID and value['code_commit'] == COMMIT
    assert value['steps'] == value['optimizer_steps'] == 2 and value['backward_finite'] and value['intended_path_observed']
    assert value['data_source']['config_data_root'].rstrip('/') == server['data_root'].rstrip('/')
    assert qref == ref('pipeline/queue.json') and rref == ref('pipeline/running.json')
    c['native_smoke_result'] = ref(out)
    evidence = [ref(out), ref(log), ref(program), ref(__file__), c['native_smoke_plan_server1']]
    c['evidence'].extend(evidence)
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='native_smoke', result='passed',
        command='python -B scripts/y84_server1_alternative.py run --token '+token,
        observation='Actual original native YAML/TRAIN two-step task on server-1 GPU0/350W; full result validation, original full evaluation/cost freeze and admission still required.', evidence=evidence)
    tick.save(ROOT/'pipeline/research_work.json', work)
    print(json.dumps(dict(result=ref(out), native_executed=True, admitted=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('operation', choices=('prepare', 'activate', 'run')); p.add_argument('--token', required=True)
    args = p.parse_args(); lease(args.token, 800 if args.operation == 'run' else 240)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try:
        {'prepare': prepare, 'activate': activate, 'run': run}[args.operation](args.token)
    finally:
        lock.close()
