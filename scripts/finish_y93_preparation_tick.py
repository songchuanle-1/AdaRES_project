"""Refresh real resources sequentially, bind Y93 preparation, and close this tick."""
import argparse
import ast
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from prepare_y93_evidence_hidden_skip import files
from retain_y93_cpu_native import HEAD, BRANCH
from y93_contract import validate_cpu
from y91_deployment import Y90, Y91, contract, git
from run_y93_native import runtime
from check_ef_execution import validate_frozen
import closure


def fresh(record, key):
    age = datetime.now().astimezone() - datetime.fromisoformat(record[key])
    assert 0 <= age.total_seconds() < 300


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == HEAD == Y91
        before_q, before_r = ref('pipeline/queue.json'), ref('pipeline/running.json')
        work = tick.load(ROOT / 'pipeline/research_work.json')
        by_id = {row['id']: row for row in work['candidates']}
        y90, y91, y92, y93 = (by_id['AdaRES-' + name] for name in ('Y90', 'Y91', 'Y92', 'Y93'))
        assert y90['stage'] == y91['stage'] == 'training'
        for candidate in (y92, y93):
            assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
            assert not candidate.get('native_smoke_result') and not candidate.get('admission_receipt')
            assert not (ROOT / ('pipeline/admissions/' + candidate['id'] + '.json')).exists()
        cpu = tick.load(closure.artifact(ROOT, y93['CPU_result']))
        validate_cpu(cpu)
        version = tick.load(closure.artifact(ROOT, y93['source_manifest']))
        assert version['parent_commit'] == version['working_HEAD'] == HEAD
        assert version['code_commit'] == y93['code_commit']
        assert version['CPU_result'] == y93['CPU_result'] and version['CPU_program'] == y93['CPU_program']
        assert version['earlier_Y92_preserved'] and version['main_index_unchanged']
        assert version['native_executed'] is False and version['admitted'] is False
        assert git('rev-parse', BRANCH).decode().strip() == y93['code_commit']
        assert git('rev-parse', y93['code_commit'] + '^').decode().strip() == HEAD
        delta = [row.split('\t') for row in git(
            'diff', '--no-renames', '--name-status', HEAD, y93['code_commit']).decode().splitlines()]
        assert sorted(delta) == sorted([['A', path.relative_to(ROOT).as_posix()] for path in files()])
        for path in files():
            assert path.read_bytes() == git('show', y93['code_commit'] + ':' + path.relative_to(ROOT).as_posix())
        prepared = tick.load(closure.artifact(ROOT, y93['native_preparation']))
        assert prepared['guard_tests_passed'] == 10 and prepared['runtime_guard_tests_passed'] == 6
        assert prepared['exact_callback_functions_and_classes'] == 15
        assert prepared['CPU_native_callbacks_actually_executed'] is True
        assert prepared['native_executed'] is False and prepared['admitted'] is False
        for key, count in [('guard_tests', 10), ('runtime_guard_tests', 6)]:
            log = closure.artifact(ROOT, prepared[key]).read_bytes()
            assert ('Ran ' + str(count) + ' tests').encode() in log and b'\nOK' in log and b'skipped' not in log
        servers = monitor.load_servers(str(ROOT))
        for name, key in [('server-1', 'native_smoke_plan_server1'), ('server-2', 'native_smoke_plan_server2_gpu0')]:
            plan = tick.load(closure.artifact(ROOT, y93[key]))
            program = closure.artifact(ROOT, plan['program']).read_text(encoding='utf-8')
            frozen = ast.literal_eval(ast.parse(program).body[0].value)
            assert validate_frozen(y93, plan, frozen) == 7
            runtime(name, y93, frozen, servers[name])
        earlier = tick.load(closure.artifact(ROOT, y92['additive_version_binding']))
        assert earlier['working_HEAD'] == earlier['parent_commit'] == HEAD
        assert earlier['guard_tests_passed'] == 15 and earlier['native_executed'] is False
        assert earlier['original_CPU_result'] == y92['CPU_result']
        assert earlier['original_CPU_program'] == y92['CPU_program']
        for dependency in earlier['evidence']:
            closure.artifact(ROOT, dependency)
        for candidate, count in [(y90, 25), (y91, 29)]:
            handoff = tick.load(closure.artifact(ROOT, candidate['terminal_handoff_plan']))
            assert handoff['CPU_guard_tests_passed'] == count
            assert handoff['checkpoint_loaded'] is False and handoff['archived'] is False
            assert handoff['post_training_freeze'] == candidate['post_training_freeze']
            for dependency in handoff['dependencies']:
                closure.artifact(ROOT, dependency)
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        assert running == {'server-1': {'0': 'AdaRES-Y90'}, 'server-2': {'0': 'AdaRES-Y91'}}
        observations, occupation = {}, {}
        for name, cid, version_, pid in [('server-1', 'AdaRES-Y90', Y90, 2442962),
                                         ('server-2', 'AdaRES-Y91', Y91, 83827)]:
            task = next(row for row in queue['gpu_queues'][name] if row['id'] == cid)
            assert task['status'] == 'running' and task['launch_verified'] is True
            assert str(task['gpu']) == '0' and task['pid'] == pid and task['code_commit'] == version_
            closure.admission(ROOT, task)
            contract(version_, HEAD, cid)
            observed = tick.load(closure.artifact(ROOT, by_id[cid]['latest_training_observation']))
            assert observed['token'] == token and observed['native_progress_confirmed'] is True
            assert observed['pid'] == pid and observed['source_marker'] == version_
            fresh(observed, 'observed_local')
            observations[cid] = dict(evidence=by_id[cid]['latest_training_observation'],
                phase=observed['after']['latest'], pid=pid)
            entry = ref('pipeline/checks/gpu_occupation_' + name + '_final_' + token + '.json')
            occupied = tick.load(closure.artifact(ROOT, entry))
            assert occupied['token'] == token and occupied['complete'] is True and occupied['read_only'] is True
            assert occupied['scope']['allowed_gpus'] == [0]
            assert all(row['rc'] == 0 for row in occupied['queries'].values())
            fresh(occupied, 'observed_at')
            occupation[name] = entry
        paused = next(row for row in queue['gpu_queues']['server-2'] if row['id'] == 'AdaRES-Y57')
        assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        assert servers['server-1']['allowed_gpus'] == servers['server-2']['allowed_gpus'] == [0]
        assert servers['server-6']['allowed_gpus'] == []
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        output = ROOT / ('pipeline/checks/y93_preparation_closeout_' + token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, recorded_at=datetime.now().astimezone().isoformat(),
            observations=observations, actual_occupation=occupation, Y93_CPU_result=y93['CPU_result'],
            Y93_retained_version=y93['source_manifest'], Y93_native_preparation=y93['native_preparation'],
            CPU_guards=10, runtime_guards=6, CPU_native_exact_callbacks=15,
            earlier_Y92_binding=y92['additive_version_binding'], original_Y90_Y91_preserved=True,
            original_full_frozen_FIRST_and_cost_preserved=True, original_GPU1_and_Y57_pause_preserved=True,
            new_scientific_candidates=1, new_admissions=0, new_training_launches=0,
            exact_original_archives=0, actual_GPU_native_tasks=0,
            actual_new_CPU_candidate_runs=1, earlier_CPU_scientific_replays=0,
            original_training_checkpoints_loaded=False, efficacy_measured=False, production_HEAD_index_queue_preserved=True,
            evidence=[ref(__file__), y93['CPU_result'], y93['CPU_program'], y93['source_manifest'],
                y93['native_preparation'], y92['additive_version_binding'], y90['terminal_handoff_plan'],
                y91['terminal_handoff_plan']] + [row['evidence'] for row in observations.values()]
                + list(occupation.values())))
        y93['evidence'].append(ref(output))
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
    closeout_path = record(token)
    closeout = tick.load(closeout_path)
    notes = tick.load(ROOT / ('pipeline/reports/notes_' + token + '.json'))
    for cid, entry in closeout['observations'].items():
        phase = entry['phase']
        notes['actions'].append(cid + ' 收尾实测：' + phase['phase'] + ' '
            + str(phase['epoch']) + '/' + str(phase['total']) + '，步数 '
            + str(phase['step']) + '/' + str(phase['count'])
            + '；原进程、来源、原生 YAML 和递增计数已核验。')
    notes['actions'].append('本轮精确准备及实时资源记录：' + closeout_path.relative_to(ROOT).as_posix())
    final_notes = ROOT / ('pipeline/reports/notes_refreshed_' + token + '.json')
    assert not final_notes.exists()
    tick.save(final_notes, notes)
    command([SKILL + '/foreground_tick.py', 'AdaRES_project', 'report', '--token', token, '--notes', str(final_notes)])
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
    print(json.dumps(dict(closeout=ref(closeout_path), observations=closeout['observations'],
        receipt_status=receipt['status'], postflight=receipt['postflight'], context_committed=True,
        lease_release=receipt['lease_release'], memory_lines=len(memory),
        archive_pending_files=0, fresh_notes=ref(final_notes)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    finish(parser.parse_args().token)
