"""Prepare and execute exact Y60 terminal archival without replaying Y59 tools."""

import argparse
import ast
from datetime import datetime
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import unittest

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from dispatch_prepared_candidates import snapshot
from check_ef_execution import validate_frozen
from finalize_y58_y59_handoff import REMOTE
import closure


CID = 'AdaRES-Y60'


def validate_task(task, candidate, paused):
    assert task['id'] == candidate['id'] == CID
    assert task['status'] == 'running' and task['launch_verified'] and int(task['gpu']) == 0
    assert int(task['pid']) > 0 and candidate['stage'] == 'training'
    assert task['config'] == candidate['config']
    assert task['code_commit'] == candidate['code_commit']
    assert paused['id'] == 'AdaRES-Y57' and paused['status'] == 'paused' and paused['dispatch_hold']


def validate_proof(proof, payload):
    assert proof['complete'] and proof['id'] == CID == payload['id']
    assert proof['process_absent'] and proof['CUDA_initialized'] is False
    assert proof['log'] == payload['log'] and re.fullmatch('[0-9a-f]{64}', proof['log_sha256'])
    assert proof['full_frozen_evaluation_executed'] is False
    checkpoints = proof['checkpoints']
    assert len(checkpoints) == 2
    directory = PurePosixPath(payload['work']) / payload['basename'] / payload['checkpoint_directory']
    for index, row in enumerate(checkpoints):
        assert PurePosixPath(row['path']) == directory / ('last.pth' if index == 0 else 'best.pth')
        assert re.fullmatch('[0-9a-f]{64}', row['sha256'])
        assert 1 <= row['epoch'] <= 30 and row['scheduler_last_epoch'] == row['epoch']
        assert row['optimizer_state_count'] > 0 and math.isfinite(row['best_miou'])
    assert checkpoints[0]['epoch'] == 30
    assert checkpoints[0]['best_miou'] == checkpoints[1]['best_miou']


def current():
    server = monitor.load_servers(str(ROOT))['server-2']
    assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
    queue = monitor.load_queue(str(ROOT))
    tasks = queue['gpu_queues']['server-2']
    task = next(t for t in tasks if t['id'] == CID)
    paused = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    validate_task(task, candidate, paused)
    closure.admission(ROOT, task)
    closure.artifact(ROOT, candidate['post_training_freeze'])
    plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    return server, queue, task, work, candidate


DISCOVER = r'''
import json,pathlib,re,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
log=pathlib.Path(P['log']).resolve()
log.relative_to(pathlib.Path(P['root'])/'logs')
last=None
with log.open('r',encoding='utf-8',errors='replace') as stream:
 for line in stream:
  paths=re.findall(r'save last pth to ([^\s]+/last\.pth)',line)
  if paths:last=paths[-1]
assert last,'No native checkpoint save announcement yet'
path=pathlib.Path(last).resolve()
path.relative_to(pathlib.Path(P['work'])/P['basename'])
assert re.fullmatch(r'\d{8}_\d{6}',path.parent.name)
assert path.is_file()
print(json.dumps(dict(complete=True,checkpoint_directory=path.parent.name,
 declared_last_path=str(path),log=str(log),checkpoint_loaded=False,remote_mutations=False)))
'''


def check(token):
    server, queue, task, work, candidate = current()
    ast.parse(REMOTE, feature_version=(3, 8))
    ast.parse(DISCOVER, feature_version=(3, 8))
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'scripts'), pattern='test_y60_terminal_handoff.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun > 0
    payload = dict(id=CID, pid=task['pid'], config=task['config'], log=task['log'],
        root=server['remote_code_root'], work=server['work_dir'], basename=Path(task['config']).stem)
    metadata = json.loads(monitor.ssh(server['host'], server['user'], shlex.quote(server['python']) +
        ' -B -c ' + shlex.quote('P=' + repr(payload) + '\n' + DISCOVER), timeout=45))
    assert metadata['complete'] and metadata['log'] == task['log']
    payload['checkpoint_directory'] = metadata['checkpoint_directory']
    output = ROOT / ('pipeline/checks/y60_terminal_handoff_' + token + '.json')
    assert not output.exists()
    tick.save(output, dict(complete=True, id=CID, observed_at=datetime.now().astimezone().isoformat(),
        payload=payload, code_commit=candidate['code_commit'], metadata=metadata,
        readonly_checkpoint_verifier=ref('scripts/finalize_y58_y59_handoff.py'),
        helper=ref(__file__), tests=ref('scripts/test_y60_terminal_handoff.py'),
        CPU_guard_tests=result.testsRun, CPU_guard_tests_passed=True,
        full_terminal_verification_executed=False, archived=False, queue_changed=False,
        native_smoke_executed=False, training_stopped=False,
        limitation='Only declared checkpoint location, source identity and CPU controls checked. '
                   'A live process is not terminal; full last30/best/optimizer/scheduler/log '
                   'verification and absent exact process are still mandatory before archival.'))
    candidate['terminal_handoff_plan'] = ref(output)
    additions = [ref(output), ref(__file__), ref('scripts/test_y60_terminal_handoff.py')]
    candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(plan=ref(output), payload=payload, CPU_guard_tests=result.testsRun,
                         archived=False, queue_changed=False), indent=2))


def finalize(token):
    server, queue, task, work, candidate = current()
    handoff = tick.load(closure.artifact(ROOT, candidate['terminal_handoff_plan']))
    assert handoff['helper'] == ref(__file__)
    closure.artifact(ROOT, handoff['readonly_checkpoint_verifier'])
    assert handoff['code_commit'] == candidate['code_commit']
    payload = handoff['payload']
    assert all(payload[k] == task[k] for k in ('id', 'pid', 'config', 'log'))
    assert payload['root'] == server['remote_code_root'] and payload['work'] == server['work_dir']
    before = snapshot(server, [task], [])
    assert before['gpu_query_rc'] == before['compute_query_rc'] == 0
    assert not before['tasks'][CID]['process_live'], 'Y60 remains live: do not stop or archive it'
    base = ROOT / ('pipeline/checks/y60_fixed30_' + token)
    assert not any(p.exists() for p in (base.with_suffix('.json'), base.with_suffix('.log')))
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        server['user'] + '@' + server['host'],
        'PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=' +
        shlex.quote(server['site_packages']) + ' ' + shlex.quote(server['python']) + ' -S -B -'],
        input=('P=' + repr(payload) + '\n' + REMOTE).encode(), capture_output=True, timeout=240)
    base.with_suffix('.log').write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0, 'Full terminal verification failed; preserve original output'
    proof = json.loads(response.stdout)
    validate_proof(proof, payload)
    after = snapshot(server, [task], [])
    assert after['gpu_query_rc'] == after['compute_query_rc'] == 0
    assert not after['tasks'][CID]['process_live']
    proof.update(observed_at=datetime.now().astimezone().isoformat(), live=after,
        source_checks=dict(exact_files=7, admission_verified=True, freeze_preserved=True),
        readonly_checkpoint_verifier=handoff['readonly_checkpoint_verifier'],
        terminal_handoff_plan=candidate['terminal_handoff_plan'],
        command='python -B scripts/finalize_y60_fixed30.py finalize --token ' + token,
        explanation='Full last30/best/model/optimizer/scheduler and terminal native log plus '
                    'absent exact process; no fabricated deathwatch or scientific promotion.')
    tick.save(base.with_suffix('.json'), proof)
    basename = payload['basename']
    destination = ROOT / ('Checkpoints/server-2/' + basename + '/' + token + '/fixed30.log')
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    subprocess.run(['scp', '-q', server['user'] + '@' + server['host'] + ':' + task['log'],
                    str(destination)], check=True, timeout=60)
    assert ref(destination)['sha256'] == proof['log_sha256']
    lease(token, 120)
    original_pull = monitor.pull_log

    def verified_pull(project_dir, host, user, remote_log, cfg_name, state, srv_name=''):
        assert Path(project_dir) == ROOT and host == server['host'] and user == server['user']
        assert remote_log == task['log'] and cfg_name == basename
        assert state == 'done_complete' and srv_name == 'server-2'
        return str(destination)

    monitor.pull_log = verified_pull
    events = []
    others = [dict(t) for t in queue['gpu_queues']['server-2'] if t['id'] != CID]
    try:
        monitor.finalize_task(str(ROOT), queue, 'server-2', server['host'], server['user'], task,
            basename, 'done_complete', proof['checkpoints'][0]['best_miou'], '30/30',
            str(30 - proof['checkpoints'][1]['epoch']), task['log'], events)
    finally:
        monitor.pull_log = original_pull
    experiment = monitor.load_experiment(str(ROOT), CID)
    experiment.update(terminal_evidence=ref(base.with_suffix('.json')),
                      terminal_checkpoint_metadata=proof['checkpoints'])
    monitor.save_experiment(str(ROOT), experiment)
    deathwatch = monitor.load_deathwatch(str(ROOT))
    deathwatch.get('server-2', {}).pop(basename, None)
    monitor.save_deathwatch(str(ROOT), deathwatch)
    additions = [ref(base.with_suffix('.json')), ref(base.with_suffix('.log')), ref(__file__)]
    candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
    candidate.update(stage='validation', scientific_status='fixed30_complete_frozen_review_pending',
        resource_work_kind='frozen_evaluation',
        next_action='Preserve fixed30 best/last/model/optimizer/scheduler and immutable '
            'post_training_freeze. Only perform registered full frozen evaluation and '
            'trained-cost comparison on exclusive authorized GPU0; no restart or scientific '
            'promotion from training logs. Advance prepared Y61 native checks when resources permit.')
    for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after',
                'unblock_condition', 'occupation_evidence'):
        candidate.pop(key, None)
    work['tick_attempt'] = dict(
        tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='resource_check', result='passed', command=proof['command'],
        observation='Authoritative completed last30/best/model/optimizer/scheduler and terminal '
                    'log plus absent exact process verified. Canonical archive completed, efficacy pending.',
        evidence=[ref(base.with_suffix('.json'))])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    assert monitor.load_queue(str(ROOT))['gpu_queues']['server-2'] == others
    assert monitor.load_running(str(ROOT)).get('server-2', {}).get('0') != CID
    print(json.dumps(dict(proof=ref(base.with_suffix('.json')), state=experiment['status'],
        checkpoints=proof['checkpoints'], events=events), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('check', 'finalize'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 360)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        (check if args.operation == 'check' else finalize)(args.token)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
