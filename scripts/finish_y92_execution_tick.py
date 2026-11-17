"""Bind new Y92 operational preparation to fresh training and release the lease."""
import argparse
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from y91_deployment import Y90, Y91, contract, git
import closure


def fresh(value, key):
    age = datetime.now().astimezone() - datetime.fromisoformat(value[key])
    assert 0 <= age.total_seconds() < 300


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y91
        before_q, before_r = ref('pipeline/queue.json'), ref('pipeline/running.json')
        work = tick.load(ROOT / 'pipeline/research_work.json')
        by_id = {row['id']: row for row in work['candidates']}
        y92, y93 = by_id['AdaRES-Y92'], by_id['AdaRES-Y93']
        for candidate in (y92, y93):
            assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
            assert not candidate.get('native_smoke_result') and not candidate.get('admission_receipt')
            assert not (ROOT / ('pipeline/admissions/' + candidate['id'] + '.json')).exists()
            for key in ('source_manifest', 'CPU_result', 'CPU_program', 'native_preparation',
                        'native_smoke_plan_server1', 'native_smoke_plan_server2_gpu0'):
                closure.artifact(ROOT, candidate[key])
        preparation = tick.load(closure.artifact(ROOT, y92['execution_preparation']))
        assert preparation['token'] == token and preparation['id'] == y92['id']
        assert preparation['source_resource_protocol_guards'] == 17
        assert preparation['scope'] == dict(server='server-1', gpu=0, power_limit_watts=350.)
        for key in ('actual_native_executed', 'actual_native_admission_guards_executed',
                    'admitted', 'queued', 'CPU_scientific_tests_replayed'):
            assert preparation[key] is False
        assert preparation['current_native_plan_activation_unchanged'] is True
        assert preparation['scientific_files_and_native_bodies_unchanged'] is True
        for entry in preparation['evidence']:
            closure.artifact(ROOT, entry)
        log = ROOT / ('pipeline/checks/y92_execution_preparation_' + token + '.log')
        rawlog = log.read_bytes()
        assert b'Ran 17 tests' in rawlog and b'\nOK' in rawlog and b'skipped' not in rawlog
        binding = tick.load(closure.artifact(ROOT, y92['additive_version_binding']))
        assert binding['parent_commit'] == binding['working_HEAD'] == Y91
        assert binding['code_commit'] == y92['code_commit'] and binding['guard_tests_passed'] == 15
        for entry in binding['evidence']:
            closure.artifact(ROOT, entry)
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        assert running == {'server-1': {'0': 'AdaRES-Y90'}, 'server-2': {'0': 'AdaRES-Y91'}}
        observations, occupations = {}, {}
        for name, cid, commit, pid in [('server-1', 'AdaRES-Y90', Y90, 2442962),
                                      ('server-2', 'AdaRES-Y91', Y91, 83827)]:
            task = next(row for row in queue['gpu_queues'][name] if row['id'] == cid)
            assert task['status'] == 'running' and task['launch_verified'] is True
            assert str(task['gpu']) == '0' and task['pid'] == pid and task['code_commit'] == commit
            closure.admission(ROOT, task)
            contract(commit, Y91, cid)
            candidate = by_id[cid]
            assert candidate['stage'] == 'training'
            handoff = tick.load(closure.artifact(ROOT, candidate['terminal_handoff_plan']))
            assert handoff['post_training_freeze'] == candidate['post_training_freeze']
            assert handoff['checkpoint_loaded'] is False and handoff['archived'] is False
            observation = tick.load(closure.artifact(ROOT, candidate['latest_training_observation']))
            assert observation['token'] == token and observation['native_progress_confirmed'] is True
            assert observation['pid'] == pid and observation['source_marker'] == commit
            fresh(observation, 'observed_local')
            observations[cid] = dict(evidence=candidate['latest_training_observation'],
                phase=observation['after']['latest'], pid=pid)
            entry = ref('pipeline/checks/gpu_occupation_' + name + '_final_' + token + '.json')
            occupation = tick.load(closure.artifact(ROOT, entry))
            assert occupation['token'] == token and occupation['complete'] is True
            assert occupation['read_only'] is True and occupation['scope']['allowed_gpus'] == [0]
            assert all(query['rc'] == 0 for query in occupation['queries'].values())
            fresh(occupation, 'observed_at')
            occupations[name] = entry
        servers = monitor.load_servers(str(ROOT))
        assert servers['server-1']['allowed_gpus'] == servers['server-2']['allowed_gpus'] == [0]
        assert servers['server-6']['allowed_gpus'] == []
        paused = next(row for row in queue['gpu_queues']['server-2'] if row['id'] == 'AdaRES-Y57')
        assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        output = ROOT / ('pipeline/checks/y92_execution_closeout_' + token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, recorded_at=datetime.now().astimezone().isoformat(),
            observations=observations, actual_occupation=occupations,
            execution_preparation=y92['execution_preparation'], source_protocol_runtime_guards=17,
            original_Y92_scientific_bytes_and_native_bodies_preserved=True,
            Y93_original_CPU_and_native_plans_preserved=True,
            original_Y90_Y91_and_existing_frozen_protocols_preserved=True,
            original_GPU1_and_Y57_pause_preserved=True,
            new_scientific_candidates=0, new_admissions=0, new_training_launches=0,
            exact_original_archives=0, actual_GPU_native_tasks=0, CPU_scientific_replays=0,
            original_training_checkpoints_loaded=False, efficacy_measured=False,
            production_HEAD_index_queue_preserved=True,
            pending_native_admission_suite_not_executed=True,
            next_action=preparation['next_action'],
            evidence=[ref(__file__), y92['execution_preparation'], y92['additive_version_binding'],
                y93['source_manifest'], y93['native_preparation'], ref(log)] +
                [entry['evidence'] for entry in observations.values()] + list(occupations.values())))
        y92['evidence'].append(ref(output))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        assert (before_q, before_r) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        return output
    finally:
        writer.close()


def finish(token):
    lease(token, 240)
    for cid in ('AdaRES-Y90', 'AdaRES-Y91'):
        command(['scripts/y91_deployment.py', 'observe', '--id', cid, '--token', token, '--label', 'final'])
    for name in ('server-1', 'server-2'):
        command([SKILL + '/gpu_occupation.py', 'AdaRES_project', '--token', token, '--server', name, '--label', 'final'])
    output = record(token)
    closeout = tick.load(output)
    notes = tick.load(ROOT / ('pipeline/reports/notes_' + token + '.json'))
    for cid, entry in closeout['observations'].items():
        phase = entry['phase']
        notes['actions'].append(cid + ' 收尾实测：' + phase['phase'] + ' '
            + str(phase['epoch']) + '/' + str(phase['total']) + '，步数 '
            + str(phase['step']) + '/' + str(phase['count'])
            + '；原进程、来源、原生 YAML、功率与递增计数已核验。')
    notes['actions'].append('本轮操作准备及真实资源记录：' + output.relative_to(ROOT).as_posix())
    notes_path = ROOT / ('pipeline/reports/notes_refreshed_' + token + '.json')
    assert not notes_path.exists()
    tick.save(notes_path, notes)
    command([SKILL + '/foreground_tick.py', 'AdaRES_project', 'report', '--token', token, '--notes', str(notes_path)])
    command([SKILL + '/foreground_tick.py', 'AdaRES_project', 'finish', '--token', token])
    receipt = tick.load(ROOT / ('pipeline/tick_receipts/' + token + '.json'))
    assert receipt['status'] == 'FINALIZED' and receipt['postflight'] == 'PASS'
    assert receipt['context_committed'] is True and receipt['lease_release'] == 'released'
    assert not (ROOT / 'pipeline/headless/.foreground.lockdir').exists()
    assert not (ROOT / 'pipeline/headless/.tick_context_state.next.json').exists()
    pending = ROOT / 'pipeline/archive_pending'
    assert not pending.exists() or not any(path.is_file() for path in pending.rglob('*'))
    memory = (ROOT / 'pipeline/codex_memory.md').read_text(encoding='utf-8').splitlines()
    assert len(memory) < 120 and sum(line.startswith('## ') for line in memory) == 7
    print(json.dumps(dict(closeout=ref(output), observations=closeout['observations'],
        receipt_status=receipt['status'], postflight=receipt['postflight'], context_committed=True,
        lease_release=receipt['lease_release'], memory_lines=len(memory), archive_pending_files=0),
        ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    finish(parser.parse_args().token)
