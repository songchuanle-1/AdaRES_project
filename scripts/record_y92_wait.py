"""Record actual bounded preparation while both current native owners continue."""
import argparse
import ast
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y92_mask_feature_quadratic import files
from y92_contract import validate_cpu
from y90_deployment import Y89, Y90, contract, git
from check_ef_execution import validate_frozen
from run_y92_native import runtime, OWNERS
import closure


def record(token):
    lease(token, 200)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        before_q, before_r = ref('pipeline/queue.json'), ref('pipeline/running.json')
        assert git('rev-parse', 'HEAD').decode().strip() == Y90
        work = tick.load(ROOT / 'pipeline/research_work.json')
        y92 = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y92')
        y91 = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y91')
        for candidate in (y91, y92):
            assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
            assert not candidate.get('native_smoke_result') and not candidate.get('admission_receipt')
            assert not (ROOT / ('pipeline/admissions/' + candidate['id'] + '.json')).exists()
        validate_cpu(tick.load(closure.artifact(ROOT, y92['CPU_result'])))
        version = tick.load(closure.artifact(ROOT, y92['source_manifest']))
        assert version['parent_commit'] == version['working_HEAD'] == Y90
        assert version['code_commit'] == y92['code_commit']
        assert version['main_index_unchanged'] and version['earlier_Y91_preserved']
        assert version['native_executed'] is False and version['admitted'] is False
        assert git('rev-parse', y92['code_commit'] + '^').decode().strip() == Y90
        delta = [row.split('\t') for row in git(
            'diff', '--no-renames', '--name-status', Y90, y92['code_commit']).decode().splitlines()]
        assert sorted(delta) == sorted([['A', path.relative_to(ROOT).as_posix()] for path in files()])
        for path in files():
            assert path.read_bytes() == git('show', y92['code_commit'] + ':' + path.relative_to(ROOT).as_posix())
        prepared = tick.load(closure.artifact(ROOT, y92['native_preparation']))
        assert prepared['guard_tests_passed'] == 10 and prepared['runtime_guard_tests_passed'] == 6
        assert prepared['exact_callback_functions'] == 13
        assert prepared['CPU_native_core_callbacks_actually_executed'] is True
        assert prepared['native_executed'] is False and prepared['admitted'] is False
        for key, count in (('guard_tests', 10), ('runtime_guard_tests', 6)):
            log = closure.artifact(ROOT, prepared[key]).read_bytes()
            assert ('Ran ' + str(count) + ' tests').encode() in log and b'skipped' not in log
        servers = monitor.load_servers(str(ROOT))
        for name, key in (('server-1', 'native_smoke_plan_server1'),
                          ('server-2', 'native_smoke_plan_server2_gpu0')):
            plan = tick.load(closure.artifact(ROOT, y92[key]))
            program = closure.artifact(ROOT, plan['program'])
            frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
            assert validate_frozen(y92, plan, frozen) == 7
            runtime(name, y92, frozen, servers[name])
        binding = tick.load(closure.artifact(ROOT, y91['additive_version_binding']))
        assert binding['code_commit'] == y91['code_commit'] and binding['parent_commit'] == Y90
        assert binding['guard_tests_passed'] == 14 and binding['matching_current_owners'] == OWNERS
        for key in ('scientific_files_identical', 'native_program_bodies_identical',
                    'only_frozen_code_commit_metadata_changed', 'original_evidence_retained',
                    'main_index_unchanged', 'other_candidates_unchanged', 'queue_unchanged'):
            assert binding[key] is True
        assert binding['native_executed'] is False and binding['CPU_scientific_experiments_replayed'] is False
        for dependency in binding['evidence']:
            closure.artifact(ROOT, dependency)
        queue = monitor.load_queue(str(ROOT))
        running = monitor.load_running(str(ROOT))
        assert running == {'server-1': {'0': 'AdaRES-Y90'}, 'server-2': {'0': 'AdaRES-Y89'}}
        observations = {}
        for name, cid, commit, pid in (('server-1', 'AdaRES-Y90', Y90, 2442962),
                                       ('server-2', 'AdaRES-Y89', Y89, 60578)):
            task = next(row for row in queue['gpu_queues'][name] if row['id'] == cid)
            assert task['status'] == 'running' and task['launch_verified'] is True and str(task['gpu']) == '0'
            assert task['pid'] == pid and task['code_commit'] == commit
            closure.admission(ROOT, task)
            contract(commit, Y90, cid)
            candidate = next(row for row in work['candidates'] if row['id'] == cid)
            observation = tick.load(closure.artifact(ROOT, candidate['latest_training_observation']))
            assert observation['token'] == token and observation['pid'] == pid
            assert observation['native_progress_confirmed'] is True and observation['source_marker'] == commit
            age = datetime.now().astimezone() - datetime.fromisoformat(observation['observed_local'])
            assert 0 <= age.total_seconds() < 300, 'Refresh actual native observation under a new label'
            observations[cid] = dict(evidence=candidate['latest_training_observation'],
                                     phase=observation['after']['latest'])
        paused = next(row for row in queue['gpu_queues']['server-2'] if row['id'] == 'AdaRES-Y57')
        assert paused['status'] == 'paused' and paused['dispatch_hold']
        assert paused['resume_requires_user_authorization'] is True
        assert servers['server-1']['allowed_gpus'] == servers['server-2']['allowed_gpus'] == [0]
        assert servers['server-6']['allowed_gpus'] == []
        for cid in ('AdaRES-Y87', 'AdaRES-Y88'):
            done = monitor.load_experiment(str(ROOT), cid)
            assert done['status'] == 'done_complete' and done.get('terminal_evidence')
            closure.artifact(ROOT, done['terminal_evidence'])
        output = ROOT / ('pipeline/checks/y92_preparation_closeout_' + token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token,
            observed_at=datetime.now().astimezone().isoformat(), actual_training=observations,
            Y91_new_identity_binding=y91['additive_version_binding'], Y91_guards=14,
            Y92_CPU_version=y92['source_manifest'], Y92_CPU_result=y92['CPU_result'],
            Y92_native_preparation=y92['native_preparation'], Y92_preparation_guards=10,
            Y92_runtime_guards=6, CPU_native_core_callbacks_identical_and_executed=13,
            new_scientific_candidates=1, new_admissions=0, new_training_launches=0, new_archives=0,
            GPU_native_executed=False, efficacy_measured=False, native_smoke_passed=False,
            prior_archives_not_replayed=True, production_HEAD_index_queue_preserved=True,
            GPU1_and_Y57_still_user_paused=True,
            original_full_frozen_FIRST_and_cost_preserved=True,
            evidence=[ref(__file__), y91['additive_version_binding'], y92['source_manifest'],
                y92['CPU_result'], y92['CPU_program'], y92['native_preparation']] +
                [entry['evidence'] for entry in observations.values()]))
        y92['evidence'].append(ref(output))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        assert (before_q, before_r) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        print(json.dumps(dict(closeout=ref(output), observations=observations,
                              new_admissions=0, native_executed=False)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    record(parser.parse_args().token)
