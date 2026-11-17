"""Prepare/test an inactive Y79 alternative in the verified private server-1 runtime."""
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

CID = 'AdaRES-Y79'
COMMIT = 'e0c7dba199fcf1a009941450d04aeb799418a7e6'
HEAD = '5eee1e50c9e1d65bf113a4b8757bfc27d028a821'
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


def read_baseline(server, token):
    relative = 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
    expected = yaml.safe_load((ROOT / 'AdaRES' / relative).read_bytes())
    remote = """import pathlib,json,subprocess,hashlib,sys
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
sys.path[:0]=SITES
import yaml
raw=pathlib.Path(FILE).read_bytes()
assert yaml.safe_load(raw)==EXPECTED
print(json.dumps(dict(complete=True,hostname='dl02-MS-7B73',raw_sha256=hashlib.sha256(raw).hexdigest(),LF_sha256=hashlib.sha256(raw.replace(b'\\r\\n',b'\\n')).hexdigest(),YAML_semantics_identical=True)))
""".replace('SITES', repr([server['site_packages']])).replace('FILE', repr(server['remote_code_root'] + '/' + relative)).replace('EXPECTED', repr(expected))
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        server['user'] + '@' + server['host'], server['python'] + ' -B -S -'],
        input=remote.encode(), capture_output=True, timeout=30)
    output = ROOT / ('pipeline/checks/y79_server1_baseline_yaml_' + token + '.json')
    assert not output.exists() and not output.with_suffix('.log').exists()
    output.with_suffix('.log').write_bytes(r.stdout + r.stderr); assert r.returncode == 0
    d = json.loads(r.stdout)
    raw = (ROOT / 'AdaRES' / relative).read_bytes()
    assert d['LF_sha256'] == hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
    d.update(local_raw=ref('AdaRES/' + relative),raw_byte_equality_required=False,
        reason='Full YAML semantics and LF-normalized bytes verified; approved deterministic runtime path mappings remain separate.',evidence=[ref(output.with_suffix('.log'))])
    tick.save(output, d)
    return ref(output)


def prepare(token):
    server = configuration(); work = tick.load(ROOT / 'pipeline/research_work.json')
    c = next(c for c in work['candidates'] if c['id'] == CID)
    assert c['stage'] == 'blocked' and c['blocker_kind'] == 'exclusive_gpu_slot' and c['code_commit'] == COMMIT
    assert not c.get('native_smoke_plan_server1') and c['required_server'] == 'server-2'
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == HEAD
    original_ref = c['native_smoke_plan_server2_gpu0']; plan = tick.load(closure.artifact(ROOT, original_ref))
    source_path = closure.artifact(ROOT, plan['program']); source = source_path.read_text(encoding='utf-8')
    frozen = ast.literal_eval(ast.parse(source).body[0].value); assert validate_frozen(c, plan, frozen) == 7
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    queue = monitor.load_queue(str(ROOT)); keeper = next(t for t in queue['gpu_queues']['server-1'] if t['id'] == 'AdaRES-Y76')
    assert keeper['status'] == 'running' and keeper['launch_verified'] is True and keeper['pid'] == 3489002
    closure.admission(ROOT, keeper)
    actual = observe(server); assert idle(actual) is False
    baseline = read_baseline(server, token)
    base = 'pipeline/checks/y79_server1_' + token
    outputs = {k: ROOT / (base + suffix) for k, suffix in (
        ('program', '_native_program.py'), ('cpu', '_cpu_program.py'), ('plan', '_plan.json'),
        ('binding', '_binding.json'), ('tests', '_guards.log'))}
    assert not any(p.exists() for p in outputs.values())
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_y79_server1_alternative.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    outputs['tests'].write_bytes(tests.stdout + tests.stderr)
    assert tests.returncode == 0 and b'Ran 8 tests' in outputs['tests'].read_bytes()
    original_frozen = copy.deepcopy(frozen)
    frozen.update(runtime_server='server-1', runtime_power_limit_watts=350.0,
        runtime_path_mappings=copy.deepcopy(server['runtime_path_mappings']),
        resource_authorization=ref('pipeline/authorizations/server1_gpu0_20261025.json'))
    closure.artifact(ROOT, frozen['resource_authorization'])
    assert frozen['variant'] == original_frozen['variant'] and frozen['code_commit'] == original_frozen['code_commit']
    original_body = source.partition('\n')[2]; body = adapted_body(original_body, 350.)
    assert body.replace(ADAPT, '').replace("== 'dl02-MS-7B73'", "== 'dl01-MS-7C60'") == original_body
    outputs['program'].write_text('FROZEN=' + repr(frozen) + '\n' + body, encoding='utf-8', newline='\n')
    # Reuse the already-passed observation callbacks; only approved runtime paths change.
    verification = tick.load(closure.artifact(ROOT, c['CPU_native_observer_verification']))
    assert verification['exact_callback_functions_and_classes'] == 16 and verification['CPU_native_callbacks_actually_executed'] is True
    original_cpu = closure.artifact(ROOT, verification['CPU_program'])
    cpu_source = original_cpu.read_text(encoding='utf-8')
    anchor = 'cfg = gorilla.Config.fromfile(str(config_path))\n'; assert cpu_source.count(anchor) == 1
    cpu_source = cpu_source.replace(anchor, anchor + ADAPT)
    metadata_anchor="CPU_native_callbacks_executed=True"
    metadata_added="implementation_test_passed=True,native_task_backward=False,runtime_data_root=cfg.data.train.data_root,runtime_pretrain=cfg.train.pretrain,CPU_native_callbacks_executed=True"
    assert cpu_source.count(metadata_anchor)==1
    cpu_source=cpu_source.replace(metadata_anchor,metadata_added)
    outputs['cpu'].write_text('FROZEN=' + repr(frozen) + '\n' + cpu_source, encoding='utf-8', newline='\n')
    assert cpu_source.replace(ADAPT, '').replace(metadata_added,metadata_anchor) == original_cpu.read_text(encoding='utf-8')
    for path in (outputs['program'], outputs['cpu']): ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    alternate = copy.deepcopy(plan)
    alternate.update(program=ref(outputs['program']), CPU_program=ref(outputs['cpu']), server='server-1', gpu=0,
        runtime_power_limit_watts=350., resource_authorization=frozen['resource_authorization'],
        runtime_path_mappings=frozen['runtime_path_mappings'], previous_plan=original_ref,
        required_resource='Fresh exclusive idle/unreserved server-1 GPU0 after original Y76 canonical fixed30 archive; exact private runtime/data/path mapping and existing350W. Preserve current Y77/Y78 and all other jobs.',
        planned_command='python -B scripts/y79_server1_alternative.py run --token <CURRENT_TOKEN>',
        activation_required=True, activated=False, native_executed=False)
    assert validate_frozen(c, alternate, frozen) == 7
    tick.save(outputs['plan'], alternate)
    tick.save(outputs['binding'], dict(complete=True, id=CID, token=token, original_plan=original_ref,
        alternative_plan=ref(outputs['plan']), original_model_and_YAML_payload_identical=True,
        code_commit_and_goal_unchanged=True, diagnostic_scientific_body_reversible_exact=True,
        CPU_observer_core_reversible_exact=True,CPU_runtime_receipt_metadata_added=True, CPU_baseline_YAML=baseline, exact_original_data=DATA_PROOF,
        runtime_library_path=server['runtime_library_path'], actual_server1_observation=actual,
        actual_idle=False, canonical_reservations=blockers(queue, monitor.load_running(str(ROOT))),
        guard_tests_passed=8, guard_log=ref(outputs['tests']), inactive=True, activated=False,
        native_executed=False, hardware_changed=False, production_sync=False, queue_changed=False,
        evidence=[ref(__file__), ref('scripts/test_y79_server1_alternative.py'), ref(original_cpu)]))
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == HEAD
    c['native_smoke_plan_server1'] = ref(outputs['plan'])
    additions = [ref(p) for p in outputs.values()] + [baseline, DATA_PROOF, ref(__file__), ref('scripts/test_y79_server1_alternative.py')]
    c['evidence'].extend(additions)
    c['next_action'] = ('Preserve all live Y76/Y77/Y78 and paused Y57. Original server-2 GPU0/1 plans stay active/alternative as recorded. '
        'A new inactive server-1 plan reuses the exact scientific body with approved host/private runtime/data/path/350W mapping. '
        'Run CPU preflight first. Only original Y76 canonical fixed30 archive plus fresh idle/unreserved GPU0 permits guarded activate then run; '
        'actual native two-step PASS remains mandatory before original full evaluation/cost freeze/admission and gated dispatch. '
        'Before production commit selection preserve/rebind all live source/observer/deployment contracts; no current sync/activation or restart.')
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(binding=ref(outputs['binding']), inactive_plan=ref(outputs['plan']), tests=8, actual_idle=False, activated=False)))


def activate(token):
    server = configuration(); work = tick.load(ROOT / 'pipeline/research_work.json'); c, plan, program, frozen = candidate_plan(work)
    assert c['stage'] == 'blocked' and c['blocker_kind'] == 'exclusive_gpu_slot'
    assert c.get('CPU_server1_runtime_preflight'), 'Real CPU runtime/data preflight is required'
    cpu = tick.load(closure.artifact(ROOT, c['CPU_server1_runtime_preflight']))
    assert cpu['complete'] and cpu['implementation_test_passed'] and not cpu['CUDA_initialized']
    assert frozen['runtime_path_mappings'] == server['runtime_path_mappings']
    done = monitor.load_experiment(str(ROOT), 'AdaRES-Y76')
    assert done and done['status'] == 'done_complete', 'Preserve original Y76 until canonical fixed30 completion'
    terminal = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
    assert terminal['completed_epochs'] == {phase:list(range(1,31)) for phase in ('Train','Val')}
    actual = observe(server); require_available(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)), actual)
    output = ROOT / ('pipeline/checks/y79_server1_activation_' + token + '.json'); assert not output.exists()
    tick.save(output, dict(complete=True, id=CID, observed_at=datetime.now().astimezone().isoformat(),
        observation=actual, previous_plan=c['native_smoke_plan'], active_plan=c['native_smoke_plan_server1'],
        no_canonical_reservation=True, original_Y76_done_complete=True, native_executed=False))
    c.update(stage='validation', required_server='server-1', required_gpu=0, native_smoke_plan=c['native_smoke_plan_server1'])
    for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition', 'occupation_evidence'): c.pop(key, None)
    c['evidence'].append(ref(output)); tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(activation=ref(output), native_executed=False)))


def probe(token, gpu):
    server = configuration(); work = tick.load(ROOT / 'pipeline/research_work.json'); c, plan, native_program, frozen = candidate_plan(work)
    assert frozen['runtime_path_mappings'] == server['runtime_path_mappings']
    closure.artifact(ROOT, frozen['resource_authorization'])
    if gpu:
        assert c['stage'] == 'validation' and c['required_server'] == 'server-1' and c['required_gpu'] == 0
        assert c['native_smoke_plan'] == c['native_smoke_plan_server1']
        require_available(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)), observe(server))
    program_path = native_program if gpu else closure.artifact(ROOT, plan['CPU_program'])
    name = 'y79_server1_smoke' if gpu else 'y79_server1_CPU_runtime'
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
        command='python -B scripts/y79_server1_alternative.py ' + ('run' if gpu else 'cpu') + ' --token ' + token,
        observation='Actual server-1 original native task two-step result; validate full protocol/admission next.' if gpu else
            'Verified private server-1 runtime with CUDA hidden constructed native-YAML full model/released backbone and real mapped original TRAIN data; exact frozen observer callbacks passed two synthetic decoder optimizer steps. No GPU/native sparse task backward/activation/admission or efficacy result.', evidence=evidence)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(result=ref(output), log=ref(log), started_at=started,
        observed_end_local=datetime.now().astimezone().isoformat(), gpu_executed=gpu,
        train_points=result.get('train_points'), train_expressions=result.get('train_expressions'),
        synthetic_decoder_optimizer_steps=result.get('synthetic_CPU_decoder_optimizer_steps'), activated=False if not gpu else True)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('operation', choices=('prepare', 'cpu', 'activate', 'run')); p.add_argument('--token', required=True); a = p.parse_args()
    lease(a.token, 800 if a.operation == 'run' else 600 if a.operation == 'cpu' else 240)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try:
        if a.operation == 'prepare': prepare(a.token)
        elif a.operation == 'activate': activate(a.token)
        else: probe(a.token, a.operation == 'run')
    finally: lock.close()


if __name__ == '__main__': main()
