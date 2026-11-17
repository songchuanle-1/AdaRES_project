"""Verify Y63 progress and archive completed Y62 without touching live training."""
import argparse
import ast
import copy
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from capture_server1_training import BODY as OBSERVER
from finalize_y61_fixed30 import REMOTE as TERMINAL, DISCOVER, validate_proof as old_validate
from check_ef_execution import validate_frozen
import closure

OLD, LIVE = 'AdaRES-Y62', 'AdaRES-Y63'


def validate_mapping(mapping):
    assert mapping.get('server-1', {}).get('0') == LIVE, 'Preserve exact live successor'
    assert all(value != OLD for slots in mapping.values() for value in slots.values())
    return copy.deepcopy(mapping)


def context():
    server = monitor.load_servers(str(ROOT))['server-1']
    assert server['host'] == '192.168.1.175' and server['allowed_gpus'] == [0]
    assert server['expected_hostname'] == 'dl02-MS-7B73'
    queue = monitor.load_queue(str(ROOT))
    tasks = {t['id']: t for t in queue['gpu_queues']['server-1']}
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidates = {c['id']: c for c in work['candidates']}
    paused = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold']
    for cid in (OLD, LIVE):
        task, candidate = tasks[cid], candidates[cid]
        assert task['status'] == 'running' and task['launch_verified'] and str(task['gpu']) == '0'
        assert task['config'] == candidate['config'] and task['code_commit'] == candidate['code_commit']
        closure.admission(ROOT, task)
        closure.artifact(ROOT, candidate['post_training_freeze'])
        plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
        program = closure.artifact(ROOT, plan['program'])
        frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(candidate, plan, frozen) == 7
    validate_mapping(monitor.load_running(str(ROOT)))
    return server, queue, tasks, work, candidates


def remote(server, program, output, cpu_torch=False):
    assert not output.exists() and not output.with_suffix('.log').exists()
    prefix = 'PYTHONDONTWRITEBYTECODE=1 '
    if cpu_torch:
        prefix += 'CUDA_VISIBLE_DEVICES= PYTHONPATH=' + shlex.quote(server['site_packages']) + ' '
    command = prefix + shlex.quote(server['python']) + (' -S' if cpu_torch else '') + ' -B -'
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        server['user'] + '@' + server['host'], command], input=program.encode(),
        capture_output=True, timeout=240 if cpu_torch else 45)
    output.with_suffix('.log').write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0, 'Read-only verification failed; preserve raw output, never relaunch'
    return json.loads(response.stdout)


def observe(token):
    server, queue, tasks, work, candidates = context()
    task = tasks[LIVE]
    cfg = yaml.safe_load((ROOT / 'AdaRES' / task['config']).read_text(encoding='utf-8'))
    def adapt(value):
        if isinstance(value, dict): return {k: adapt(v) for k, v in value.items()}
        if isinstance(value, list): return [adapt(v) for v in value]
        if isinstance(value, str):
            for mapping in server['runtime_path_mappings']:
                old, new = mapping['source'].rstrip('/'), mapping['target'].rstrip('/')
                if value == old or value.startswith(old + '/'): value = new + value[len(old):]
        return value
    folder = ROOT / 'AdaRES/ipdn/model_paper5/adares_y63_peer_bilinear'
    files = {p.relative_to(ROOT / 'AdaRES').as_posix(): ref(p)['sha256'] for p in folder.glob('*.py')}
    assert len(files) == 6
    payload = dict(root=server['remote_code_root'], task=task, source_files=files,
                   expected_config=adapt(cfg))
    body = OBSERVER.replace("training_efficacy='Unknown; no validation result yet.'",
                            "training_efficacy='Unknown; full frozen evaluation not executed.'")
    body = body.replace('print(json.dumps(dict(complete=True',
        "inventory=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader']).decode().strip()\n"
        "owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']).decode()\n"
        "assert [int(row.split(',')[1]) for row in owners.splitlines() if row.split(',')[0].strip()==inventory]==[int(task['pid'])]\n"
        "assert (root/'.codex_source_commit').read_text().strip()==task['code_commit']\n"
        "print(json.dumps(dict(complete=True,compute_owners=owners,authorized_gpu_uuid=inventory")
    ast.parse(body, feature_version=(3, 8))
    output = ROOT / ('pipeline/checks/y63_server1_progress_' + token + '.json')
    result = remote(server, 'P=' + repr(payload) + '\n' + body, output)
    result.update(id=LIVE, token=token, observed_at=datetime.now().astimezone().isoformat(),
        source_files=files, command='python -B scripts/reconcile_y62_y63_server1.py observe --token ' + token,
        evidence=[ref(__file__), ref(output.with_suffix('.log'))])
    tick.save(output, result)
    candidate = candidates[LIVE]
    candidate.update(stage='training', latest_training_observation=ref(output),
        next_action='Preserve exact running server-1 GPU0 Y63 fixed30/seed1999. Observe native progress; '
                    'only after actual completion verify last/best/optimizer/scheduler and execute frozen evaluation.')
    candidate['evidence'].append(ref(output))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(output), before=result['before'], after=result['after'], gpu=result['gpu'])))


def finalize(token):
    server, queue, tasks, work, candidates = context()
    old, live = tasks[OLD], tasks[LIVE]
    candidate = candidates[OLD]
    assert candidate['stage'] == 'training' and candidates[LIVE]['stage'] == 'training'
    observation = tick.load(closure.artifact(ROOT, candidates[LIVE]['latest_training_observation']))
    assert observation['token'] == token and observation['pid'] == live['pid']
    assert (datetime.now().astimezone() - datetime.fromisoformat(observation['observed_at'])).total_seconds() < 900
    payload = dict(id=OLD, pid=old['pid'], config=old['config'], log=old['log'],
        root=server['remote_code_root'], work=server['work_dir'], basename=Path(old['config']).stem)
    discover_path = ROOT / ('pipeline/checks/y62_server1_location_' + token + '.json')
    location = remote(server, 'P=' + repr(payload) + '\n' + DISCOVER.replace('dl01-MS-7C60', 'dl02-MS-7B73'), discover_path)
    tick.save(discover_path, location)
    payload['checkpoint_directory'] = location['checkpoint_directory']
    output = ROOT / ('pipeline/checks/y62_server1_fixed30_' + token + '.json')
    body = TERMINAL.replace('dl01-MS-7C60', 'dl02-MS-7B73')
    proof = remote(server, 'P=' + repr(payload) + '\n' + body, output, cpu_torch=True)
    # The established validator is CID-bound: use its source unchanged except exact ID.
    import finalize_y61_fixed30 as established
    original_cid = established.CID
    try:
        established.CID = OLD
        old_validate(proof, payload)
    finally:
        established.CID = original_cid
    proof.update(observed_at=datetime.now().astimezone().isoformat(), token=token,
        source_checks=dict(exact_files=7, admission_verified=True, freeze_preserved=True),
        live_observation=candidates[LIVE]['latest_training_observation'],
        command='python -B scripts/reconcile_y62_y63_server1.py finalize --token ' + token,
        evidence=[ref(__file__), ref(discover_path), ref(output.with_suffix('.log'))])
    tick.save(output, proof)
    destination = ROOT / ('Checkpoints/server-1/' + payload['basename'] + '/' + token + '/fixed30.log')
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    subprocess.run(['scp', '-q', server['user'] + '@' + server['host'] + ':' + old['log'],
                    str(destination)], check=True, timeout=60)
    assert ref(destination)['sha256'] == proof['log_sha256']
    lease(token, 120)
    before_queue = copy.deepcopy(queue)
    running_before = monitor.load_running(str(ROOT))
    validate_mapping(running_before)
    pull = monitor.pull_log
    def verified_pull(project_dir, host, user, remote_log, cfg_name, state, srv_name=''):
        assert Path(project_dir) == ROOT and host == server['host'] and user == server['user']
        assert remote_log == old['log'] and cfg_name == payload['basename']
        assert state == 'done_complete' and srv_name == 'server-1'
        return str(destination)
    monitor.pull_log = verified_pull
    events = []
    try:
        monitor.finalize_task(str(ROOT), queue, 'server-1', server['host'], server['user'], old,
            payload['basename'], 'done_complete', proof['checkpoints'][0]['best_miou'], '30/30',
            gpu=old['gpu'], remote_log=old['log'], events=events)
    finally:
        monitor.pull_log = pull
    experiment = monitor.load_experiment(str(ROOT), OLD)
    experiment.update(terminal_evidence=ref(output), terminal_checkpoint_metadata=proof['checkpoints'])
    monitor.save_experiment(str(ROOT), experiment)
    deathwatch = monitor.load_deathwatch(str(ROOT))
    deathwatch.get('server-1', {}).pop(payload['basename'], None)
    monitor.save_deathwatch(str(ROOT), deathwatch)
    expected = copy.deepcopy(before_queue)
    expected['gpu_queues']['server-1'] = [t for t in expected['gpu_queues']['server-1'] if t['id'] != OLD]
    assert monitor.load_queue(str(ROOT)) == expected
    assert monitor.load_running(str(ROOT)) == running_before
    validate_mapping(monitor.load_running(str(ROOT)))
    candidate.update(stage='validation', scientific_status='fixed30_complete_frozen_review_pending',
        resource_work_kind='frozen_evaluation', required_server='server-1', terminal_evidence=ref(output),
        next_action='Preserve verified fixed30 last/best/model/optimizer/scheduler and immutable freeze. '
                    'Run only registered full frozen evaluation/cost comparison on an exclusive authorized GPU; '
                    'server-1 GPU0 currently runs Y63. No restart or efficacy promotion from training best.')
    candidate['evidence'].extend([ref(output), ref(output.with_suffix('.log')), ref(__file__)])
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=OLD, kind='resource_check', result='passed', command=proof['command'],
        observation='Verified fixed30 last/best/model/optimizer/scheduler and absent exact Y62 process; '
                    'canonical terminal archive preserved live Y63, running mapping and paused Y57. Efficacy pending.',
        evidence=[ref(output)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(proof=ref(output), checkpoints=proof['checkpoints'], events=events,
                         live_Y63_and_pause_preserved=True)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('observe', 'finalize'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 360)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        (observe if args.operation == 'observe' else finalize)(args.token)
    finally:
        lock.close()


if __name__ == '__main__': main()
