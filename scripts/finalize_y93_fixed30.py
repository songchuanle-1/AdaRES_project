"""Close the exact completed server-2 Y93 run; preserve every other task."""
import argparse
import ast
import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import shlex
from reconcile_y62_y63_server1 import remote
import finalize_y61_fixed30 as established
import closure
CID = 'AdaRES-Y93'
DISCOVER = established.DISCOVER.replace('dl01-MS-7C60', 'dl01-MS-7C60')
REMOTE = established.REMOTE.replace('dl01-MS-7C60', 'dl01-MS-7C60')
REMOTE = REMOTE.replace('raw=log.read_bytes()', 'log_stat=(log.stat().st_size,log.stat().st_mtime_ns)\nraw=log.read_bytes()')
REMOTE = REMOTE.replace('announced=re.findall', "completed_epochs={phase:sorted(set(int(x) for x in re.findall(phase+r'\\[(\\d+)/30\\]:[^\\r\\n]*?'+count+r'/'+count+r'\\s*\\[',text))) for phase,count in [('Train','4686'),('Val','363')]}\nassert all(value==list(range(1,31)) for value in completed_epochs.values())\nannounced=re.findall")
REMOTE = REMOTE.replace('metrics=re.findall', 'assert log_stat==(log.stat().st_size,log.stat().st_mtime_ns)\nmetrics=re.findall')
REMOTE = REMOTE.replace('final_logged_metrics=', 'completed_epochs=completed_epochs,final_logged_metrics=')
from y93_terminal_contract import CHECKPOINT_DIRECTORY, IDENTITY, build_contract
REMOTE = REMOTE.replace('import torch', IDENTITY + '\nimport torch')
RESOURCE = "import datetime,json,pathlib,subprocess\nassert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'\nassert not pathlib.Path('/proc/'+str(P['pid'])).exists()\ngpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu,power.limit','--format=csv,noheader,nounits']).decode().strip()\ncompute=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()\nprint(json.dumps(dict(complete=True,process_absent=True,hostname='dl01-MS-7C60',gpu=gpu,compute=compute,observed_at=datetime.datetime.now().astimezone().isoformat())))\n"

def validate_task(task, candidate, paused):
    assert task['id'] == candidate['id'] == CID
    assert task['status'] == 'running' and task['launch_verified'] and (str(task['gpu']) == '0')
    assert task['server'] == 'server-2' and task['pid'] == 47380
    assert candidate['stage'] == 'training'
    assert task['config'] == candidate['config'] == 'configs_song/model_paper5/adares_y93_evidence_hidden_skip_res.yaml'
    assert task['code_commit'] == candidate['code_commit'] == 'daa93b4375f144f0e79da92f85fe1fe27e877e18'
    assert paused['id'] == 'AdaRES-Y57' and paused['status'] == 'paused' and paused['dispatch_hold']
    assert paused.get('resume_requires_user_authorization') is True

def expected_mapping(running):
    assert running.get('server-2', {}).get('0') == CID, 'Do not remove another slot owner'
    assert sum((v == CID for slots in running.values() for v in slots.values())) == 1
    expected = copy.deepcopy(running)
    del expected['server-2']['0']
    if not expected['server-2']:
        del expected['server-2']
    return expected

def validate_proof(proof, payload):
    assert payload['checkpoint_directory'] == CHECKPOINT_DIRECTORY
    original = established.CID
    try:
        established.CID = CID
        established.validate_proof(proof, payload)
        assert proof['completed_epochs'] == {p: list(range(1, 31)) for p in ('Train', 'Val')}
    finally:
        established.CID = original

def validate_resource(observation):
    import math
    assert observation['complete'] is True and observation['process_absent'] is True
    assert observation['hostname'] == 'dl01-MS-7C60'
    used, util, power = [float(v.strip()) for v in observation['gpu'].split(',')]
    assert all((math.isfinite(v) for v in (used, util, power)))
    assert used >= 0 and 0 <= util <= 100 and (power == 280.0)
    assert isinstance(observation['compute'], str)
    return used < 1024 and util < 10 and (not observation['compute'].strip())

def current():
    server = monitor.load_servers(str(ROOT))['server-2']
    closure.validate_adares_server(ROOT, server)
    queue = monitor.load_queue(str(ROOT))
    task = next((t for t in queue['gpu_queues']['server-2'] if t['id'] == CID))
    paused = next((t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y57'))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next((c for c in work['candidates'] if c['id'] == CID))
    validate_task(task, candidate, paused)
    assert task['code_commit'] == 'daa93b4375f144f0e79da92f85fe1fe27e877e18'
    assert ref('pipeline/admissions/AdaRES-Y93.json')['sha256'] == 'f6b87f2813f1e8e429772bc0f63963fe1a43e0582c6c65e21771092dcbf69832'
    closure.admission(ROOT, task)
    assert candidate['post_training_freeze']['sha256'] == '8563a67a55b91f7c72b2899d03917b28d322e2f5ac14130c016511c72b0c7ef8'
    closure.artifact(ROOT, candidate['post_training_freeze'])
    plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    return (server, queue, task, work, candidate)

def validate_location(location, payload):
    from pathlib import PurePosixPath
    assert location['complete'] is True and location['checkpoint_loaded'] is False
    assert location['remote_mutations'] is False and location['log'] == payload['log']
    assert location['checkpoint_directory'] == CHECKPOINT_DIRECTORY
    expected = PurePosixPath(payload['work']) / payload['basename'] / CHECKPOINT_DIRECTORY / 'last.pth'
    assert PurePosixPath(location['declared_last_path']) == expected

def validate_archived_experiment(experiment, task):
    assert task['id'] == CID and task['server'] == 'server-2' and (str(task['gpu']) == '0')
    assert experiment['id'] == task['id'] and experiment['status'] == 'done_complete'
    assert str(experiment['gpu']) == str(task['gpu'])

def evaluation_resource(task):
    assert task['id'] == CID and task['server'] == 'server-2' and (str(task['gpu']) == '0')
    return dict(required_server=task['server'], required_gpu=int(task['gpu']))

def finalize(token):
    server, queue, task, work, candidate = current()
    before_queue = copy.deepcopy(queue)
    before_running = monitor.load_running(str(ROOT))
    mapping = expected_mapping(before_running)
    assert not any((t['id'] != CID and t['status'] in ('running', 'claiming') and (str(t.get('gpu')) == '0') for t in queue['gpu_queues']['server-2']))
    base = ROOT / ('pipeline/checks/y93_server2_fixed30_' + token)
    location_path = base.with_name(base.name + '_location.json')
    resource_path = base.with_name(base.name + '_resource.json')
    binding_path = base.with_name(base.name + '_archive_binding.json')
    tests_path = base.with_name(base.name + '_tests.log')
    assert not any((p.exists() for p in (base.with_suffix('.json'), base.with_suffix('.log'), location_path, resource_path, binding_path, tests_path)))
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts', '-p', 'test_y93_terminal_handoff.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    tests_path.write_bytes(tests.stdout + tests.stderr)
    assert tests.returncode == 0 and b'Ran 29 tests' in tests_path.read_bytes()
    config = yaml.safe_load((ROOT / 'AdaRES' / task['config']).read_text(encoding='utf-8'))

    def adapt(value):
        if isinstance(value, dict):
            return {k: adapt(v) for k, v in value.items()}
        if isinstance(value, list):
            return [adapt(v) for v in value]
        if isinstance(value, str):
            for m in server.get('runtime_path_mappings', []):
                old, new = (m['source'].rstrip('/'), m['target'].rstrip('/'))
                if value == old or value.startswith(old + '/'):
                    value = new + value[len(old):]
        return value
    folder = ROOT / 'AdaRES/ipdn/model_paper5/adares_y93_evidence_hidden_skip'
    source_files = {p.relative_to(ROOT / 'AdaRES').as_posix(): ref(p)['sha256'] for p in folder.glob('*.py')}
    assert len(source_files) == 6
    payload = dict(id=CID, pid=task['pid'], config=task['config'], log=task['log'], root=server['remote_code_root'], work=server['work_dir'], basename=Path(task['config']).stem, source_files=source_files, expected_config=adapt(config), code_commit=task['code_commit'])
    payload.update(build_contract(task, server))
    handoff = tick.load(closure.artifact(ROOT, candidate['terminal_handoff_plan']))
    assert handoff['id'] == CID and handoff['complete'] is True
    assert handoff['payload'] == payload and handoff['CPU_guard_tests_passed'] == 29
    assert handoff['post_training_freeze'] == candidate['post_training_freeze']
    assert handoff['admission'] == ref('pipeline/admissions/AdaRES-Y93.json')
    for dependency in handoff['dependencies']:
        closure.artifact(ROOT, dependency)
    validate_location(tick.load(closure.artifact(ROOT, handoff['declared_location'])), payload)
    pre_path = base.with_name(base.name + '_before.json')
    before = remote(server, 'P=' + repr(payload) + '\n' + RESOURCE, pre_path)
    tick.save(pre_path, before)
    discovery = DISCOVER.replace('print(json.dumps(dict(complete=True', IDENTITY + 'print(json.dumps(dict(complete=True')
    location = remote(server, 'P=' + repr(payload) + '\n' + discovery, location_path)
    validate_location(location, payload)
    tick.save(location_path, location)
    payload['checkpoint_directory'] = location['checkpoint_directory']
    proof = remote(server, 'P=' + repr(payload) + '\n' + REMOTE, base.with_suffix('.json'), cpu_torch=True)
    validate_proof(proof, payload)
    after = remote(server, 'P=' + repr(payload) + '\n' + RESOURCE, resource_path)
    tick.save(resource_path, after)
    idle = validate_resource(after)
    proof.update(observed_at=datetime.now().astimezone().isoformat(), token=token, source_checks=dict(exact_model_files=6, native_YAML_semantics_verified=True, source_marker_verified=True, admission_verified=True, freeze_preserved=True), resource_after=ref(resource_path), CPU_guard_tests_passed=29, command='python -B scripts/finalize_y93_fixed30.py --token ' + token, evidence=[ref(__file__), ref(location_path), ref(base.with_suffix('.log')), ref(tests_path), ref(pre_path)])
    tick.save(base.with_suffix('.json'), proof)
    destination = ROOT / ('Checkpoints/server-2/' + payload['basename'] + '/' + token + '/fixed30.log')
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    subprocess.run(['scp', '-q', server['user'] + '@' + server['host'] + ':' + task['log'], str(destination)], check=True, timeout=60)
    assert ref(destination)['sha256'] == proof['log_sha256']
    lease(token, 120)
    assert monitor.load_queue(str(ROOT)) == before_queue and monitor.load_running(str(ROOT)) == before_running
    pull = monitor.pull_log

    def verified_pull(project_dir, host, user, remote_log, cfg_name, state, srv_name=''):
        assert Path(project_dir) == ROOT and host == server['host'] and (user == server['user'])
        assert remote_log == task['log'] and cfg_name == payload['basename']
        assert state == 'done_complete' and srv_name == 'server-2'
        return str(destination)
    monitor.pull_log = verified_pull
    events = []
    try:
        monitor.finalize_task(str(ROOT), queue, 'server-2', server['host'], server['user'], task, payload['basename'], 'done_complete', proof['checkpoints'][0]['best_miou'], '30/30', gpu=task['gpu'], remote_log=task['log'], events=events)
    finally:
        monitor.pull_log = pull
    expected_queue = copy.deepcopy(before_queue)
    expected_queue['gpu_queues']['server-2'] = [t for t in expected_queue['gpu_queues']['server-2'] if t['id'] != CID]
    assert monitor.load_queue(str(ROOT)) == expected_queue
    assert monitor.load_running(str(ROOT)) == mapping
    experiment = monitor.load_experiment(str(ROOT), CID)
    validate_archived_experiment(experiment, task)
    experiment.update(terminal_evidence=ref(base.with_suffix('.json')), terminal_checkpoint_metadata=proof['checkpoints'])
    monitor.save_experiment(str(ROOT), experiment)
    deathwatch = monitor.load_deathwatch(str(ROOT))
    deathwatch.get('server-2', {}).pop(payload['basename'], None)
    monitor.save_deathwatch(str(ROOT), deathwatch)
    candidate.update(stage='validation', scientific_status='fixed30_complete_frozen_review_pending', resource_work_kind='frozen_evaluation', **evaluation_resource(task), terminal_evidence=ref(base.with_suffix('.json')), next_action='Preserve verified fixed30 last/best/model/optimizer/scheduler and immutable freeze. Only run the original full frozen evaluation/cost comparison on exclusive authorized GPU0. No restart or scientific promotion from training logs; advance prepared candidates only after their exact server-2 private-runtime/path/280W native plan is verified.')
    for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition', 'occupation_evidence'):
        candidate.pop(key, None)
    if not idle:
        candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot', blocker='Exact Y93 process exited and fixed30 verified; physical server-2 GPU0 still has foreign compute or non-idle occupation. Preserve those jobs; frozen evaluation waits.', retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(), unblock_condition='Fresh exclusive authorized idle server-2 GPU0 with no compute PID or canonical reservation.')
    tick.save(binding_path, dict(complete=True, token=token, id=CID, terminal_evidence=ref(base.with_suffix('.json')), before_running=before_running, after_running=mapping, only_exact_Y89_archived=True, paused_Y57_and_all_other_queue_entries_preserved=True, remote_mutations=False, checkpoint_files_retained=True, full_frozen_evaluation_executed=False, resource_after=ref(resource_path), physical_idle_observed=idle, helper=ref(__file__)))
    candidate['evidence'].extend([ref(base.with_suffix('.json')), ref(base.with_suffix('.log')), ref(binding_path), ref(resource_path), ref(resource_path.with_suffix('.log')), ref(__file__), ref(tests_path), ref('scripts/test_y93_terminal_handoff.py'), ref('scripts/y93_terminal_contract.py')])
    candidate['terminal_archive_binding'] = ref(binding_path)
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'], candidate_id=CID, kind='resource_check', result='passed', command=proof['command'], observation='Exact Y93 absent; native fixed30 log and stable finite last/best/model/optimizer/scheduler verified. Canonical archive completed with explicit GPU0; paused Y57/foreign jobs preserved. Efficacy pending.', evidence=[ref(base.with_suffix('.json')), ref(binding_path)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(proof=ref(base.with_suffix('.json')), binding=ref(binding_path), checkpoints=proof['checkpoints'], physical_idle=idle, resource=after, events=events, native_GPU_steps_executed=False, efficacy_unknown=True)))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--token', required=True)
    a = p.parse_args()
    lease(a.token, 600)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        finalize(a.token)
    finally:
        writer.close()
