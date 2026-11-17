"""Bind the real Y87 terminal, Y89 launch and preserved Y88 handoff evidence."""
import argparse
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from advance_y89 import prepared, validate_result, Y89, GRANT, GRANT_SHA
from run_y89_native import validate_terminal
import closure


def record(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate, _, _, frozen = prepared(work)
        assert candidate['stage'] == 'training' and closure.git(ROOT, 'rev-parse', 'HEAD') == Y89
        validate_result(tick.load(closure.artifact(ROOT, candidate['native_smoke_result'])), candidate, frozen)
        queue = monitor.load_queue(str(ROOT))
        task = next(row for row in queue['gpu_queues']['server-2'] if row['id'] == 'AdaRES-Y89')
        assert task['status'] == 'running' and task['launch_verified'] is True
        assert (task['server'], str(task['gpu']), task['code_commit']) == ('server-2', '0', Y89)
        assert task['pid'] == 60578
        closure.admission(ROOT, task)
        original = monitor.load_experiment(str(ROOT), 'AdaRES-Y87')
        assert original['status'] == 'done_complete' and str(original['gpu']) == '0'
        terminal = tick.load(closure.artifact(ROOT, original['terminal_evidence']))
        validate_terminal(terminal, 'server-2')
        y87 = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y87')
        assert y87['stage'] == 'blocked' and y87['blocker_kind'] == 'exclusive_gpu_slot'
        y88 = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y88')
        assert y88['stage'] == 'training'
        binding = tick.load(closure.artifact(ROOT, y88['terminal_handoff_plan']))
        assert binding['only_tested_additive_source_binding'] is True
        assert binding['CPU_guard_tests_passed'] == 25 and binding['additive_guard_tests_passed'] == 6
        assert binding['supersedes'] == y88['terminal_handoff_plan_before_y89']
        for entry in binding['original_helpers_unchanged']:
            closure.artifact(ROOT, entry)
        observations = {}
        for row in (candidate, y88):
            entry = row['latest_training_observation']
            observation = tick.load(closure.artifact(ROOT, entry))
            age = (datetime.now().astimezone() - datetime.fromisoformat(observation['observed_local'])).total_seconds()
            assert 0 <= age < 300 and observation['native_progress_confirmed'] is True
            observations[row['id']] = entry
        guards = ROOT / ('pipeline/checks/y89_execution_guards_' + token + '.log')
        admission_guards = ROOT / ('pipeline/checks/adares-y89_admission_guards_' + token + '.log')
        assert b'Ran 13 tests' in guards.read_bytes() and b'OK' in guards.read_bytes()
        assert b'Ran 22 tests' in admission_guards.read_bytes() and b'OK' in admission_guards.read_bytes()
        assert ref(GRANT)['sha256'] == GRANT_SHA
        helpers = [ref('scripts/' + name) for name in ('advance_y89.py', 'y89_deployment.py',
            'test_y89_execution_guards.py', 'test_y89_admission.py', 'record_y89_execution.py')]
        selection = ref('pipeline/checks/adares-y89_source_selection_' + token + '.json')
        selected = tick.load(closure.artifact(ROOT, selection))
        assert selected['target'] == Y89 and selected['other_index_entries_preserved'] is True
        dispatch = ref('pipeline/checks/adares-y89_dispatch_' + token + '.json')
        launched = tick.load(closure.artifact(ROOT, dispatch))
        assert launched['task']['pid'] == task['pid'] and launched['other_queue_entries_preserved'] is True
        evidence = helpers + [selection, dispatch, ref(guards), ref(admission_guards),
            original['terminal_evidence'], y87['terminal_archive_binding'], y88['terminal_handoff_plan'],
            ref('pipeline/checks/y88_y89_binding_test_repair_' + token + '.json')] + list(observations.values())
        out = ROOT / ('pipeline/checks/y89_execution_closeout_' + token + '.json')
        assert not out.exists()
        tick.save(out, dict(complete=True, token=token, id='AdaRES-Y89',
            observed_at=datetime.now().astimezone().isoformat(), code_commit=Y89,
            native_smoke=candidate['native_smoke_result'], admission=ref(candidate['admission_receipt']),
            full_frozen_review=candidate['post_training_freeze'], source_selection=selection, dispatch=dispatch,
            original_Y87_terminal=original['terminal_evidence'], original_Y88_handoff=y88['terminal_handoff_plan'],
            latest_training_observations=observations, execution_guards=13, actual_native_admission_guards=22,
            new_scientific_candidates=0, admissions=1, launches=1, archives=1, actual_native_optimizer_steps=2,
            formal_FIRST_updated=False, efficacy_unknown=True, GPU1_used=False, Y57_resumed=False,
            unrelated_index_and_queue_entries_preserved=True, evidence=evidence))
        candidate['execution_closeout'] = ref(out)
        candidate['evidence'].extend([ref(out)] + evidence)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(closeout=ref(out), admissions=1, launches=1, archives=1,
                              training_observations=observations, formal_FIRST_updated=False)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    record(parser.parse_args().token)
