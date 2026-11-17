"""Bind this tick's Y92 preparation and Y91 terminal plan to fresh observations."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from rebind_y92_on_y91 import OLD, BRANCH, PLAN_KEYS, source_delta
from rebind_y90_on_y89 import parse_program, compare_programs
from check_ef_execution import validate_frozen
from run_y92_bound_native import OWNERS, runtime, validate_binding
from y91_deployment import Y90, Y91, contract, git
import closure


def fresh(value, field):
    age = datetime.now().astimezone() - datetime.fromisoformat(value[field])
    assert 0 <= age.total_seconds() < 300, 'Refresh the actual observation under a new label'


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y91
        before_q, before_r = ref('pipeline/queue.json'), ref('pipeline/running.json')
        work = tick.load(ROOT / 'pipeline/research_work.json')
        by_id = {row['id']: row for row in work['candidates']}
        y90, y91, y92 = (by_id['AdaRES-' + name] for name in ('Y90', 'Y91', 'Y92'))
        assert y90['stage'] == y91['stage'] == 'training'
        assert y92['stage'] == 'blocked' and y92['blocker_kind'] == 'exclusive_gpu_slot'
        assert not y92.get('native_smoke_result') and not y92.get('admission_receipt')
        assert not (ROOT / 'pipeline/admissions/AdaRES-Y92.json').exists()
        binding = tick.load(closure.artifact(ROOT, y92['additive_version_binding']))
        validate_binding(binding, y92, Y91)
        assert binding['token'] == token and binding['guard_tests_passed'] == 15
        assert binding['original_commit'] == OLD and binding['retained_branch'] == BRANCH
        assert binding['native_executed'] is False and binding['admitted'] is False
        assert binding['CPU_scientific_experiments_replayed'] is False
        assert binding['scientific_revision'] is False and binding['production_sync'] is False
        for dependency in binding['evidence']:
            closure.artifact(ROOT, dependency)
        assert git('rev-parse', BRANCH).decode().strip() == y92['code_commit']
        source_delta(Y91, y92['code_commit'], [row[1] for row in binding['exact_delta']])
        proposal = tick.load(ROOT / ('pipeline/checks/y92_additive_binding_' + token + '_proposal.json'))
        assert hashlib.sha256(Path(proposal['main_index']).read_bytes()).hexdigest() == proposal['main_index_sha256']
        assert (proposal['queue'], proposal['running']) == (before_q, before_r)
        assert y92['CPU_result'] == binding['original_CPU_result']
        assert y92['CPU_program'] == binding['original_CPU_program']
        assert y92['native_preparation'] == binding['original_native_preparation']
        for label, count in [('y92_additive_binding_', 15), ('y91_terminal_preparation_', 29)]:
            suffix = '.log' if count == 15 else '_tests.log'
            log = (ROOT / ('pipeline/checks/' + label + token + suffix)).read_bytes()
            assert ('Ran ' + str(count) + ' tests').encode() in log and b'\nOK' in log
            assert b'skipped' not in log and b'FAILED' not in log
        servers = monitor.load_servers(str(ROOT))
        for name, key in PLAN_KEYS.items():
            plan = tick.load(closure.artifact(ROOT, y92[key]))
            original = tick.load(closure.artifact(ROOT, binding['original_plans'][name]))
            frozen = compare_programs(original['program'], plan['program'], y92, OLD)
            assert validate_frozen(y92, plan, frozen) == 7
            runtime(name, y92, frozen, servers[name])
        handoff = tick.load(closure.artifact(ROOT, y91['terminal_handoff_plan']))
        assert handoff['token'] == token and handoff['id'] == 'AdaRES-Y91'
        assert handoff['CPU_guard_tests_passed'] == 29 and handoff['complete'] is True
        assert handoff['post_training_freeze'] == y91['post_training_freeze']
        assert handoff['admission'] == ref('pipeline/admissions/AdaRES-Y91.json')
        for field in ('full_terminal_verification_executed', 'checkpoint_loaded', 'archived',
                      'training_stopped', 'queue_changed', 'remote_mutations'):
            assert handoff[field] is False
        location = tick.load(closure.artifact(ROOT, handoff['declared_location']))
        assert location['checkpoint_directory'] == '20261107_102124'
        assert location['checkpoint_loaded'] is False and location['remote_mutations'] is False
        for dependency in handoff['dependencies']:
            closure.artifact(ROOT, dependency)
        prior = tick.load(closure.artifact(ROOT, y90['terminal_handoff_plan']))
        assert prior['CPU_guard_tests_passed'] == 25 and prior['additive_guard_tests_passed'] == 6
        assert prior['checkpoint_loaded'] is False and prior['archived'] is False
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        assert running == {'server-1': {'0': 'AdaRES-Y90'}, 'server-2': {'0': 'AdaRES-Y91'}}
        observations, occupation = {}, {}
        for name, cid in OWNERS.items():
            task = next(row for row in queue['gpu_queues'][name] if row['id'] == cid)
            version, pid = (Y90, 2442962) if name == 'server-1' else (Y91, 83827)
            assert task['status'] == 'running' and task['launch_verified'] is True
            assert str(task['gpu']) == '0' and task['pid'] == pid and task['code_commit'] == version
            closure.admission(ROOT, task)
            contract(version, Y91, cid)
            candidate = by_id[cid]
            observed = tick.load(closure.artifact(ROOT, candidate['latest_training_observation']))
            assert observed['token'] == token and observed['native_progress_confirmed'] is True
            assert observed['pid'] == pid and observed['source_marker'] == version
            fresh(observed, 'observed_local')
            observations[cid] = dict(evidence=candidate['latest_training_observation'],
                phase=observed['after']['latest'], pid=pid)
            entry = ref('pipeline/checks/gpu_occupation_' + name + '_final_' + token + '.json')
            occupied = tick.load(closure.artifact(ROOT, entry))
            assert occupied['complete'] is True and occupied['read_only'] is True
            assert occupied['token'] == token and occupied['scope']['allowed_gpus'] == [0]
            assert all(row['rc'] == 0 for row in occupied['queries'].values())
            fresh(occupied, 'observed_at')
            occupation[name] = entry
        paused = next(row for row in queue['gpu_queues']['server-2'] if row['id'] == 'AdaRES-Y57')
        assert paused['status'] == 'paused' and paused['dispatch_hold']
        assert paused['resume_requires_user_authorization'] is True
        assert servers['server-1']['allowed_gpus'] == servers['server-2']['allowed_gpus'] == [0]
        assert servers['server-6']['allowed_gpus'] == []
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        output = ROOT / ('pipeline/checks/y92_binding_closeout_' + token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token,
            recorded_at=datetime.now().astimezone().isoformat(), observations=observations,
            actual_current_occupation=occupation, Y92_additive_binding=y92['additive_version_binding'],
            Y92_binding_guards_passed=15, Y91_terminal_handoff=y91['terminal_handoff_plan'],
            Y91_terminal_guards_passed=29, exact_Y91_declared_directory='20261107_102124',
            production_HEAD_and_index_unchanged=True, queue_and_original_training_preserved=True,
            original_scientific_inputs_and_native_bodies_identical=True,
            original_CPU_scientific_experiments_not_replayed=True,
            GPU1_and_Y57_still_user_paused=True, original_full_frozen_FIRST_and_cost_preserved=True,
            new_scientific_candidates=0, new_admissions=0, new_training_launches=0,
            actual_GPU_native_tasks=0, exact_original_archives=0,
            checkpoint_loaded=False, efficacy_measured=False, scientific_revision=False,
            evidence=[ref(__file__), y92['additive_version_binding'], y91['terminal_handoff_plan'],
                y90['terminal_handoff_plan'], handoff['declared_location']] +
                [row['evidence'] for row in observations.values()] + list(occupation.values())))
        y91['evidence'].append(ref(output))
        y92['evidence'].append(ref(output))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        assert (before_q, before_r) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        print(json.dumps(dict(closeout=ref(output), observations=observations,
                              occupation=occupation, native_executed=False, admitted=False)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    record(parser.parse_args().token)
