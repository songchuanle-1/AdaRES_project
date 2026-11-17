"""Record actual Y88 archive, Y90 launch and untouched Y89/Y91 boundaries."""
import argparse
import copy
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from advance_y90 import make_freeze, prepared, validate_result
from run_y90_bound_native import validate_terminal
from y90_deployment import Y88, Y89, Y90, contract, git
import closure


def record(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y90
        before_q, before_r = ref('pipeline/queue.json'), ref('pipeline/running.json')
        work = tick.load(ROOT / 'pipeline/research_work.json')
        y90, _, _, frozen = prepared(work)
        assert y90['stage'] == 'training' and y90['code_commit'] == Y90
        raw = tick.load(closure.artifact(ROOT, y90['native_smoke_result']))
        validate_result(raw, y90, frozen)
        receipt = ref(y90['admission_receipt'])
        assert receipt == ref('pipeline/admissions/AdaRES-Y90.json')
        native = tick.load(closure.artifact(ROOT, y90['native_smoke']))
        assert native['losses'] == raw['losses'] and native['server'] == 'server-1'
        review = tick.load(closure.artifact(ROOT, y90['post_training_freeze']))
        y88 = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y88')
        parent = tick.load(closure.artifact(ROOT, y88['post_training_freeze']))
        expected = make_freeze(parent, y88['post_training_freeze'])
        assert review['evaluation'] == expected['evaluation'] and review['training'] == parent['training']
        assert review['parent_protocol'] == y88['post_training_freeze']
        assert review['resource_condition'] == expected['resource_condition']
        done = monitor.load_experiment(str(ROOT), 'AdaRES-Y88')
        assert done['status'] == 'done_complete' and str(done['gpu']) == '0'
        terminal = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
        validate_terminal(terminal, 'server-1')
        assert terminal['checkpoints'][0]['best_miou'] == .5037736296653748
        assert '/20261106_053553/' in terminal['checkpoints'][0]['path']
        archive = tick.load(closure.artifact(ROOT, y88['terminal_archive_binding']))
        assert archive['only_exact_Y88_archived'] and archive['checkpoint_files_retained']
        assert archive['remote_mutations'] is False and archive['full_frozen_evaluation_executed'] is False
        assert y88['stage'] == 'blocked' and y88['blocker_kind'] == 'exclusive_gpu_slot'
        queue = monitor.load_queue(str(ROOT))
        running = monitor.load_running(str(ROOT))
        expected_running = {'server-1': {'0': 'AdaRES-Y90'}, 'server-2': {'0': 'AdaRES-Y89'}}
        assert running == expected_running
        for name, cid, version, pid in (('server-1', 'AdaRES-Y90', Y90, 2442962),
                                        ('server-2', 'AdaRES-Y89', Y89, 60578)):
            task = next(row for row in queue['gpu_queues'][name] if row['id'] == cid)
            assert task['status'] == 'running' and task['launch_verified'] is True
            assert str(task['gpu']) == '0' and task['pid'] == pid and task['code_commit'] == version
            closure.admission(ROOT, task)
            candidate = next(row for row in work['candidates'] if row['id'] == cid)
            observed = tick.load(closure.artifact(ROOT, candidate['latest_training_observation']))
            assert observed['token'] == token and observed['native_progress_confirmed'] is True
            assert observed['pid'] == pid and observed['source_marker'] == version
            age = datetime.now().astimezone() - datetime.fromisoformat(observed['observed_local'])
            assert 0 <= age.total_seconds() < 300, 'Collect a fresh labeled observation before closeout'
        paused = next(row for row in queue['gpu_queues']['server-2'] if row['id'] == 'AdaRES-Y57')
        assert paused['status'] == 'paused' and paused['dispatch_hold']
        assert paused['resume_requires_user_authorization'] is True
        assert set(contract(Y89, Y90, 'AdaRES-Y89')) == {Y89, Y90}
        y89 = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y89')
        handoff = tick.load(closure.artifact(ROOT, y89['terminal_handoff_plan']))
        assert handoff['CPU_guard_tests_passed'] == 25 and handoff['checkpoint_loaded'] is False
        assert handoff['archived'] is False and handoff['remote_mutations'] is False
        assert handoff['post_training_freeze'] == y89['post_training_freeze']
        location = tick.load(closure.artifact(ROOT, handoff['declared_location']))
        assert location['checkpoint_directory'] == '20261106_122357'
        assert location['checkpoint_loaded'] is False and location['remote_mutations'] is False
        for dependency in handoff['dependencies']:
            closure.artifact(ROOT, dependency)
        guards = ROOT / ('pipeline/checks/adares-y90_admission_guards_' + token + '.log')
        assert b'Ran 23 tests' in guards.read_bytes() and b'skipped' not in guards.read_bytes()
        selection = ROOT / ('pipeline/checks/adares-y90_source_selection_' + token + '.json')
        selected = tick.load(selection)
        assert selected['parent'] == Y89 and selected['target'] == Y90
        assert selected['other_index_entries_preserved'] and selected['working_files_modified'] is False
        dependencies = [done['terminal_evidence'], y88['terminal_archive_binding'], receipt,
            y90['native_smoke_result'], y90['native_smoke'], y90['post_training_freeze'],
            y90['latest_training_observation'], y89['latest_training_observation'],
            y89['terminal_handoff_plan'], ref(selection), ref(guards), ref(__file__),
            ref('pipeline/checks/y89_terminal_helpers_' + token + '.json'),
            ref('pipeline/checks/y90_execution_preparation_' + token + '.json')]
        output = ROOT / ('pipeline/checks/y90_execution_closeout_' + token + '.json')
        assert not output.exists()
        y91 = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y91')
        assert y91['stage'] == 'blocked' and y91['blocker_kind'] == 'exclusive_gpu_slot'
        assert not y91.get('native_smoke_result') and not y91.get('admission_receipt')
        original_y91 = {key: copy.deepcopy(y91[key]) for key in ('code_commit', 'CPU_result',
            'CPU_program', 'native_preparation', 'native_smoke_plan_server1', 'native_smoke_plan_server2_gpu0')}
        y91.update(
            blocker='Current Y90/server1 and original Y89/server2 reserve both authorized GPU0 resources. '
                'Y91 retains its original CPU-tested branch and native plans; no current-head native PASS.',
            next_action='Do not replay completed CPU tests/version/native preparation. Before GPU-native use, '
                'make a new explicitly tested additive identity/runtime/owner binding of the SAME seven Y91 '
                'files onto current Y90, preserving all original evidence and old plans/helpers. Only after '
                'matching Y90/server1 or Y89/server2 full fixed30 checkpoint verification/canonical archive '
                'and fresh exclusive idle/unreserved authorized GPU0 execute the newly bound native plan. '
                'No GPU1/Y57 use, scientific tuning, source reset or old-runner bypass.')
        assert all(y91[key] == value for key, value in original_y91.items())
        tick.save(output, dict(complete=True, token=token,
            observed_at=datetime.now().astimezone().isoformat(),
            Y88_original_fixed30_checkpoint_verified_and_archived=True,
            Y88_training_best_not_FIRST=terminal['checkpoints'][0]['best_miou'],
            Y90_real_native_losses=raw['losses'], Y90_native_peak_reserved_bytes=raw['peak_reserved_bytes'],
            Y90_actual_native_admission_guards=23, Y90_source_resource_protocol_guards=13,
            Y90_actual_training_PID=2442962, increasing_native_counters_verified=True,
            Y89_PID_and_original_marker_preserved=True, Y89_terminal_preparation_guards=25,
            Y89_checkpoint_loaded=False, Y91_original_preparation_preserved=original_y91,
            new_admissions=1, new_training_launches=1, exact_archives=1, new_scientific_candidates=0,
            formal_FIRST_evaluation_executed=False, cost_comparison_executed=False,
            GPU1_and_Y57_still_user_paused=True, queue_running_unchanged=True,
            evidence=dependencies))
        y90['evidence'].append(ref(output))
        y91['evidence'].append(ref(output))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        assert (before_q, before_r) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        print(json.dumps(dict(closeout=ref(output), new_admissions=1, new_launches=1,
                              archived='AdaRES-Y88', formal_FIRST_updated=False)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    record(parser.parse_args().token)
