"""Close the exact completed server-2 Y81 run; preserve every other task."""
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

CID = 'AdaRES-Y81'
DISCOVER = established.DISCOVER.replace('dl01-MS-7C60', 'dl01-MS-7C60')
REMOTE = established.REMOTE.replace('dl01-MS-7C60', 'dl01-MS-7C60')
REMOTE = REMOTE.replace("raw=log.read_bytes()", "log_stat=(log.stat().st_size,log.stat().st_mtime_ns)\nraw=log.read_bytes()")
REMOTE = REMOTE.replace("announced=re.findall", "completed_epochs={phase:sorted(set(int(x) for x in re.findall(phase+r'\\[(\\d+)/30\\]:[^\\r\\n]*?'+count+r'/'+count+r'\\s*\\[',text))) for phase,count in [('Train','4686'),('Val','363')]}\nassert all(value==list(range(1,31)) for value in completed_epochs.values())\nannounced=re.findall")
REMOTE = REMOTE.replace("metrics=re.findall", "assert log_stat==(log.stat().st_size,log.stat().st_mtime_ns)\nmetrics=re.findall")
REMOTE = REMOTE.replace("final_logged_metrics=", "completed_epochs=completed_epochs,final_logged_metrics=")

from y81_terminal_contract import CHECKPOINT_DIRECTORY, IDENTITY, build_contract
# Recheck source, deterministic YAML mappings and the retained deployment marker
# in the same read-only process that will verify the stable checkpoints.
REMOTE = REMOTE.replace('import torch', IDENTITY + '\nimport torch')
RESOURCE = """import datetime,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
assert not pathlib.Path('/proc/'+str(P['pid'])).exists()
processes=subprocess.check_output(['ps','-u','songcl','-o','pid=,args=']).decode()
assert not any('tools/train.py' in line and P['config'] in line for line in processes.splitlines())
print(json.dumps(dict(complete=True,process_absent=True,same_config_process_absent=True,hostname='dl01-MS-7C60',GPU1_use_authorized=False,GPU_queries_executed=False,remote_mutations=False,observed_at=datetime.datetime.now().astimezone().isoformat())))
"""


def validate_task(task, candidate, paused):
    assert task['id'] == candidate['id'] == CID
    assert task['status'] == 'running' and task['launch_verified'] and str(task['gpu']) == '1'
    assert task['server'] == 'server-2' and task['pid'] == 120624
    assert candidate['stage'] == 'training'
    assert task['config'] == candidate['config'] == 'configs_song/model_paper5/adares_y81_query_visual_feedback_res.yaml'
    assert task['code_commit'] == candidate['code_commit'] == '4157937d0c3cf344a44394ce15c95349aa94d50f'
    assert paused['id'] == 'AdaRES-Y57' and paused['status'] == 'paused' and paused['dispatch_hold']
    assert paused.get('resume_requires_user_authorization') is True


def expected_mapping(running):
    assert running.get('server-2', {}).get('1') == CID, 'Do not remove another slot owner'
    assert sum(v == CID for slots in running.values() for v in slots.values()) == 1
    expected = copy.deepcopy(running)
    del expected['server-2']['1']
    if not expected['server-2']:
        del expected['server-2']
    return expected


def validate_proof(proof, payload):
    assert payload['checkpoint_directory'] == CHECKPOINT_DIRECTORY
    original = established.CID
    try:
        established.CID = CID
        established.validate_proof(proof, payload)
        assert proof['completed_epochs'] == {p:list(range(1,31)) for p in ('Train','Val')}
    finally:
        established.CID = original


def validate_resource(observation):
    assert observation['complete'] is True and observation['process_absent'] is True
    assert observation['same_config_process_absent'] is True
    assert observation['hostname'] == 'dl01-MS-7C60'
    assert observation['GPU1_use_authorized'] is False
    assert observation['GPU_queries_executed'] is False and observation['remote_mutations'] is False
    return False  # Explicit user pause; physical idleness cannot authorize use.


def current():
    server = monitor.load_servers(str(ROOT))['server-2']
    closure.validate_adares_server(ROOT, server)
    queue = monitor.load_queue(str(ROOT))
    task = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == CID)
    paused = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y57')
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    validate_task(task, candidate, paused)
    assert task['code_commit'] == '4157937d0c3cf344a44394ce15c95349aa94d50f'
    assert ref('pipeline/admissions/AdaRES-Y81.json')['sha256'] == '76f21c9d98991ae8bc7f41cb6e3ad41bfc96301f2caf4be222161bbe7c8f2648'
    assert server['allowed_gpus'] == [0]
    pause_authority = tick.load(closure.artifact(ROOT, task['resource_pause']))
    assert pause_authority['resource_use_paused'] and pause_authority['paused_gpu'] == 1
    assert pause_authority['resource_resume_requires_user_authorization']
    assert candidate['post_training_freeze']['sha256'] == 'b7e6407383e14330fff75b3fcc66f553510df9b479794084dde6cffde1c31ed4'
    closure.admission(ROOT, task)
    closure.artifact(ROOT, candidate['post_training_freeze'])
    plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    return server, queue, task, work, candidate


def validate_location(location, payload):
    from pathlib import PurePosixPath
    assert location['complete'] is True and location['checkpoint_loaded'] is False
    assert location['remote_mutations'] is False and location['log'] == payload['log']
    assert location['checkpoint_directory'] == CHECKPOINT_DIRECTORY
    expected = PurePosixPath(payload['work']) / payload['basename'] / CHECKPOINT_DIRECTORY / 'last.pth'
    assert PurePosixPath(location['declared_last_path']) == expected


def validate_archived_experiment(experiment, task):
    assert task['id'] == CID and task['server'] == 'server-2' and str(task['gpu']) == '1'
    assert experiment['id'] == task['id'] and experiment['status'] == 'done_complete'
    assert str(experiment['gpu']) == str(task['gpu'])


def evaluation_resource(task):
    assert task['id'] == CID and task['server'] == 'server-2' and str(task['gpu']) == '1'
    return dict(required_server=task['server'], required_gpu=int(task['gpu']))


def finalize(token):
    server, queue, task, work, candidate = current()
    before_queue = copy.deepcopy(queue)
    before_running = monitor.load_running(str(ROOT))
    mapping = expected_mapping(before_running)
    assert not any(t['id'] != CID and t['status'] in ('running', 'claiming') and str(t.get('gpu')) == '1'
                   for t in queue['gpu_queues']['server-2'])
    base = ROOT / ('pipeline/checks/y81_server2_fixed30_' + token)
    location_path = base.with_name(base.name + '_location.json')
    resource_path = base.with_name(base.name + '_resource.json')
    binding_path = base.with_name(base.name + '_archive_binding.json')
    tests_path = base.with_name(base.name + '_tests.log')
    assert not any(p.exists() for p in (base.with_suffix('.json'), base.with_suffix('.log'),
        location_path, resource_path, binding_path, tests_path))
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_y81_terminal_handoff.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    tests_path.write_bytes(tests.stdout + tests.stderr)
    assert tests.returncode == 0 and b'Ran 25 tests' in tests_path.read_bytes()
    config = yaml.safe_load((ROOT / 'AdaRES' / task['config']).read_text(encoding='utf-8'))
    def adapt(value):
        if isinstance(value, dict): return {k: adapt(v) for k, v in value.items()}
        if isinstance(value, list): return [adapt(v) for v in value]
        if isinstance(value, str):
            for m in server.get('runtime_path_mappings', []):
                old, new = m['source'].rstrip('/'), m['target'].rstrip('/')
                if value == old or value.startswith(old + '/'): value = new + value[len(old):]
        return value
    folder = ROOT / 'AdaRES/ipdn/model_paper5/adares_y81_query_visual_feedback'
    source_files = {p.relative_to(ROOT / 'AdaRES').as_posix(): ref(p)['sha256'] for p in folder.glob('*.py')}
    assert len(source_files) == 6
    payload = dict(id=CID, pid=task['pid'], config=task['config'], log=task['log'],
        root=server['remote_code_root'], work=server['work_dir'], basename=Path(task['config']).stem,
        source_files=source_files, expected_config=adapt(config), code_commit=task['code_commit'])
    payload.update(build_contract(task, server))
    handoff = tick.load(closure.artifact(ROOT, candidate['terminal_handoff_plan']))
    assert handoff['id'] == CID and handoff['complete'] is True
    assert handoff['payload'] == payload and handoff['CPU_guard_tests_passed'] == 25
    assert handoff['post_training_freeze'] == candidate['post_training_freeze']
    assert handoff['admission'] == ref('pipeline/admissions/AdaRES-Y81.json')
    for dependency in handoff['dependencies']:
        closure.artifact(ROOT, dependency)
    validate_location(tick.load(closure.artifact(ROOT, handoff['declared_location'])), payload)
    # Refuse a still-live exact PID before loading either checkpoint.
    pre_path = base.with_name(base.name + '_before.json')
    before = remote(server, 'P=' + repr(payload) + '\n' + RESOURCE, pre_path)
    tick.save(pre_path, before)
    discovery = DISCOVER.replace("print(json.dumps(dict(complete=True", IDENTITY + "print(json.dumps(dict(complete=True")
    location = remote(server, 'P=' + repr(payload) + '\n' + discovery, location_path)
    validate_location(location, payload)
    tick.save(location_path, location)
    payload['checkpoint_directory'] = location['checkpoint_directory']
    proof = remote(server, 'P=' + repr(payload) + '\n' + REMOTE, base.with_suffix('.json'), cpu_torch=True)
    validate_proof(proof, payload)
    after = remote(server, 'P=' + repr(payload) + '\n' + RESOURCE, resource_path)
    tick.save(resource_path, after)
    idle = validate_resource(after)
    proof.update(observed_at=datetime.now().astimezone().isoformat(), token=token,
        source_checks=dict(exact_model_files=6, native_YAML_semantics_verified=True,
            source_marker_verified=True, admission_verified=True, freeze_preserved=True),
        resource_after=ref(resource_path), CPU_guard_tests_passed=25,
        command='python -B scripts/finalize_y81_fixed30.py --token ' + token,
        evidence=[ref(__file__), ref(location_path), ref(base.with_suffix('.log')), ref(tests_path), ref(pre_path)])
    tick.save(base.with_suffix('.json'), proof)
    destination = ROOT / ('Checkpoints/server-2/' + payload['basename'] + '/' + token + '/fixed30.log')
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    subprocess.run(['scp', '-q', server['user'] + '@' + server['host'] + ':' + task['log'],
        str(destination)], check=True, timeout=60)
    assert ref(destination)['sha256'] == proof['log_sha256']
    lease(token, 120)
    assert monitor.load_queue(str(ROOT)) == before_queue and monitor.load_running(str(ROOT)) == before_running
    pull = monitor.pull_log
    def verified_pull(project_dir, host, user, remote_log, cfg_name, state, srv_name=''):
        assert Path(project_dir) == ROOT and host == server['host'] and user == server['user']
        assert remote_log == task['log'] and cfg_name == payload['basename']
        assert state == 'done_complete' and srv_name == 'server-2'
        return str(destination)
    monitor.pull_log = verified_pull
    events = []
    try:
        monitor.finalize_task(str(ROOT), queue, 'server-2', server['host'], server['user'], task,
            payload['basename'], 'done_complete', proof['checkpoints'][0]['best_miou'], '30/30',
            gpu=task['gpu'], remote_log=task['log'], events=events)
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
    candidate.update(stage='validation', scientific_status='fixed30_complete_frozen_review_pending',
        resource_work_kind='frozen_evaluation', **evaluation_resource(task),
        terminal_evidence=ref(base.with_suffix('.json')),
        next_action='Preserve verified fixed30 last/best/model/optimizer/scheduler and immutable freeze. '
            'GPU1 is user-paused; original complete evaluation/cost remains frozen on server-2 GPU1/390W until explicit resource reauthorization and an exclusive idle slot. '
            'No restart, migration, resource substitution or scientific promotion from training logs. Independent prepared GPU0 work may continue.')
    for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition', 'occupation_evidence'):
        candidate.pop(key, None)
    if not idle:
        candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='human_authorization',
            blocker='The user paused all server-2 GPU1 use on 2026-11-04. Preserve the original complete frozen evaluation/cost on GPU1/390W; no test, training, evaluation or migration.',
            retry_after=(datetime.now().astimezone() + timedelta(hours=4)).isoformat(),
            unblock_condition='Explicit user GPU1 resource reauthorization, followed by fresh exclusive original GPU1/390W ownership. A timer or idle card is not authorization.',
            resource_pause=task['resource_pause'])
    tick.save(binding_path, dict(complete=True, token=token, id=CID, terminal_evidence=ref(base.with_suffix('.json')),
        before_running=before_running, after_running=mapping, only_exact_Y81_archived=True,
        paused_Y57_and_all_other_queue_entries_preserved=True, remote_mutations=False,
        checkpoint_files_retained=True, full_frozen_evaluation_executed=False,
        resource_after=ref(resource_path), physical_idle_observed=None, GPU1_use_paused=True, CPU_only_terminal_verification=True, helper=ref(__file__)))
    candidate['evidence'].extend([ref(base.with_suffix('.json')), ref(base.with_suffix('.log')),
        ref(binding_path), ref(resource_path), ref(resource_path.with_suffix('.log')), ref(__file__),
        ref(tests_path), ref('scripts/test_y81_terminal_handoff.py'), ref('scripts/y81_terminal_contract.py'), task['resource_pause']])
    candidate['terminal_archive_binding'] = ref(binding_path)
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='resource_check', result='passed', command=proof['command'],
        observation='Exact Y81 absent; native fixed30 log and stable finite last/best/model/optimizer/scheduler '
            'verified. Canonical archive completed with explicit original GPU1; current GPU1 pause retained; paused Y57/foreign jobs preserved. Efficacy pending.',
        evidence=[ref(base.with_suffix('.json')), ref(binding_path)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(proof=ref(base.with_suffix('.json')), binding=ref(binding_path), checkpoints=proof['checkpoints'],
        GPU1_use_paused=True, resource=after, events=events, native_GPU_steps_executed=False, efficacy_unknown=True)))


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
