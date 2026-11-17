"""Bind identical CPU-tested Y93 inputs to current source and GPU0 owners."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import unittest
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from retain_y91_cpu_native import git
from check_ef_execution import validate_frozen
from y93_contract import validate_cpu
from y92_deployment import Y91, Y92, contract
from rebind_y90_on_y89 import parse_program, compare_programs, runner_module
import closure
CID = 'AdaRES-Y93'
OLD = '7a9f1aa67e054d79b28c6af3160a5b1cc11a51c9'
OLD_BRANCH = 'refs/heads/adares-y93-evidence-hidden-skip'
BRANCH = 'refs/heads/adares-y93-evidence-hidden-skip-on-y92'
RUNNER = 'scripts/run_y93_bound_native_v2.py'
PLAN_KEYS = {'server-1': 'native_smoke_plan_server1', 'server-2': 'native_smoke_plan_server2_gpu0'}
OWNERS = {'server-1': 'AdaRES-Y92', 'server-2': 'AdaRES-Y91'}

def source_delta(parent, commit, paths):
    assert git('rev-parse', commit + '^').decode().strip() == parent
    rows = [row.split('\t') for row in git('diff', '--no-renames', '--name-status', parent, commit).decode().splitlines()]
    assert len(rows) == 7 and all((len(row) == 2 and row[0] == 'A' for row in rows))
    assert {row[1] for row in rows} == set(paths)
    for path in paths:
        assert (ROOT / path).read_bytes() == git('show', OLD + ':' + path) == git('show', commit + ':' + path)
    return rows

def parent_evidence():
    from advance_y92 import prepared
    from y92_contract import validate_result
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate, _, _, frozen = prepared(work)
    entry = candidate['native_smoke_result']
    validate_result(tick.load(closure.artifact(ROOT, entry)), candidate, frozen)
    task = next((row for row in monitor.load_queue(str(ROOT))['gpu_queues']['server-1'] if row['id'] == 'AdaRES-Y92'))
    assert task['status'] == 'running' and task['launch_verified'] is True
    closure.admission(ROOT, task)
    return [entry, ref(candidate['admission_receipt']), candidate['post_training_freeze']]

class BindingGuards(unittest.TestCase):
    proposal = None

    def setUp(self):
        self.p = self.proposal
        self.c = self.p['candidate']

    def test_exact_seven_previously_tested_additions(self):
        source_delta(Y92, self.c['code_commit'], self.p['paths'])

    def test_production_HEAD_and_unrelated_index_unchanged(self):
        self.assertEqual(git('rev-parse', 'HEAD').decode().strip(), Y92)
        self.assertEqual(hashlib.sha256(Path(self.p['main_index']).read_bytes()).hexdigest(), self.p['main_index_sha256'])

    def test_original_branch_and_CPU_evidence_retained(self):
        self.assertEqual(git('rev-parse', OLD_BRANCH).decode().strip(), OLD)
        self.assertEqual(self.c['CPU_result'], self.p['original']['CPU_result'])
        self.assertEqual(self.c['CPU_program'], self.p['original']['CPU_program'])
        validate_cpu(tick.load(closure.artifact(ROOT, self.c['CPU_result'])))
        closure.artifact(ROOT, self.p['original']['source_manifest'])

    def test_both_scientific_payloads_and_native_bodies_identical(self):
        for name in PLAN_KEYS:
            new = tick.load(ROOT / self.p['plans'][name]['path'])
            old = tick.load(closure.artifact(ROOT, self.p['original'][PLAN_KEYS[name]]))
            value = compare_programs(old['program'], new['program'], self.c, OLD)
            self.assertEqual(validate_frozen(self.c, new, value), 7)

    def test_stale_goal_tampered_payload_and_wrong_commit_refused(self):
        for name in PLAN_KEYS:
            plan = tick.load(ROOT / self.p['plans'][name]['path'])
            frozen, _ = parse_program(plan['program'])
            for field in ('goal', 'source', 'commit'):
                changed = copy.deepcopy(frozen)
                if field == 'goal':
                    changed['goal_sha256'] = '0' * 64
                elif field == 'source':
                    changed['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
                else:
                    changed['code_commit'] = OLD
                with self.subTest(server=name, field=field), self.assertRaises(AssertionError):
                    validate_frozen(self.c, plan, changed)

    def test_current_Y91_Y92_sources_preserved(self):
        self.assertEqual(set(contract(Y91, Y92, 'AdaRES-Y91')), {Y91, Y92})
        self.assertEqual(set(contract(Y92, Y92, 'AdaRES-Y92')), {Y92})

    def test_current_whitelists_and_pause_unchanged(self):
        servers = monitor.load_servers(str(ROOT))
        self.assertEqual(servers['server-1']['allowed_gpus'], [0])
        self.assertEqual(servers['server-2']['allowed_gpus'], [0])
        self.assertEqual(servers['server-6']['allowed_gpus'], [])
        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'], '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')

    def test_native_runtime_alternatives_and_power(self):
        module = runner_module(ROOT / RUNNER)
        servers = monitor.load_servers(str(ROOT))
        for name in PLAN_KEYS:
            plan = tick.load(ROOT / self.p['plans'][name]['path'])
            frozen, _ = parse_program(plan['program'])
            self.assertEqual(module.runtime(name, self.c, frozen, servers[name]), 350.0 if name == 'server-1' else 280.0)
            for key, value in [('runtime_physical_gpu', 1), ('runtime_power_limit_watts', 390.0), ('runtime_server', 'server-6')]:
                with self.assertRaises(AssertionError):
                    module.runtime(name, self.c, dict(frozen, **{key: value}), servers[name])
            with self.assertRaises(AssertionError):
                module.runtime(name, self.c, frozen, dict(servers[name], expected_hostname='foreign'))

    def test_runner_requires_matching_current_owner_full_completion(self):
        module = runner_module(ROOT / RUNNER)
        self.assertEqual(module.OWNERS, OWNERS)
        for name, owner in OWNERS.items():
            proof = dict(id=owner, complete=True, process_absent=True, CUDA_initialized=False, completed_epochs={phase: list(range(1, 31)) for phase in ('Train', 'Val')}, checkpoints=[dict(epoch=epoch, scheduler_last_epoch=epoch, optimizer_state_count=1, model_finite=True, optimizer_finite=True) for epoch in (30, 21)])
            module.validate_terminal(proof, name)
            for key, value in [('id', 'AdaRES-Y89'), ('id', 'AdaRES-Y88'), ('process_absent', False), ('CUDA_initialized', True)]:
                with self.assertRaises(AssertionError):
                    module.validate_terminal(dict(proof, **{key: value}), name)
            for field, value in [('scheduler_last_epoch', 29), ('model_finite', False), ('optimizer_finite', False), ('optimizer_state_count', 0)]:
                changed = copy.deepcopy(proof)
                changed['checkpoints'][0][field] = value
                with self.assertRaises(AssertionError):
                    module.validate_terminal(changed, name)
            proof['completed_epochs']['Train'].remove(12)
            with self.assertRaises(AssertionError):
                module.validate_terminal(proof, name)

    def test_actual_binding_rejects_stale_HEAD_owner_and_identity(self):
        module = runner_module(ROOT / RUNNER)
        binding = dict(complete=True, id=CID, code_commit=self.c['code_commit'], parent_commit=Y92, working_HEAD=Y92, matching_current_owners=OWNERS, scientific_files_identical=True, native_program_bodies_identical=True, only_frozen_code_commit_metadata_changed=True)
        module.validate_binding(binding, self.c, Y92)
        for key, value in [('complete', False), ('id', 'AdaRES-Y94'), ('code_commit', OLD), ('parent_commit', Y91), ('working_HEAD', Y91), ('matching_current_owners', {'server-1': 'AdaRES-Y91', 'server-2': 'AdaRES-Y89'}), ('scientific_files_identical', False), ('native_program_bodies_identical', False), ('only_frozen_code_commit_metadata_changed', False)]:
            with self.subTest(field=key), self.assertRaises(AssertionError):
                module.validate_binding(dict(binding, **{key: value}), self.c, Y92)
        with self.assertRaises(AssertionError):
            module.validate_binding(binding, self.c, Y91)

    def test_original_plan_resource_and_path_mappings_retained(self):
        for name in PLAN_KEYS:
            old = tick.load(closure.artifact(ROOT, self.p['original'][PLAN_KEYS[name]]))
            new = tick.load(ROOT / self.p['plans'][name]['path'])
            for key in ('id', 'config', 'goal_sha256', 'server', 'gpu', 'runtime_power_limit_watts', 'resource_authorization'):
                self.assertEqual(new[key], old[key])
            self.assertFalse(new['native_executed'])
            self.assertFalse(new['admitted'])
            self.assertEqual(new['execution_helper'], ref(RUNNER))

    def test_binding_and_owner_checked_before_remote_mutation(self):
        source = (ROOT / RUNNER).read_text(encoding='utf-8')
        self.assertLess(source.index("candidate['additive_version_binding']"), source.index('observation = live(server, 0)'))
        self.assertLess(source.index('validate_binding(binding, candidate, closure.git('), source.index('observation = live(server, 0)'))
        self.assertLess(source.index('validate_terminal(terminal, name)'), source.index('observation = live(server, 0)'))
        self.assertIn("assert plan['execution_helper'] == ref(__file__)", source)
        ast.parse(source, feature_version=(3, 8))

    def test_all_old_inputs_and_other_candidates_unchanged(self):
        for entry in self.p['original_inputs']:
            closure.artifact(ROOT, entry)
        current = tick.load(ROOT / 'pipeline/research_work.json')
        self.assertEqual([row for row in current['candidates'] if row['id'] != CID], self.p['other_candidates'])
        self.assertEqual(self.p['queue'], ref('pipeline/queue.json'))
        self.assertEqual(self.p['running'], ref('pipeline/running.json'))

    def test_current_reservations_cannot_be_replaced_by_old_archive(self):
        module = runner_module(ROOT / RUNNER)
        queue = monitor.load_queue(str(ROOT))
        running = monitor.load_running(str(ROOT))
        for name, owner in OWNERS.items():
            self.assertEqual(running[name]['0'], owner)
            task = next((row for row in queue['gpu_queues'][name] if row['id'] == owner))
            self.assertEqual(task['status'], 'running')
            self.assertTrue(task['launch_verified'])
            with self.assertRaises(AssertionError):
                module.unreserved(queue, running, name, 0)

    def test_original_runner_and_immutable_helpers_retained(self):
        for entry in self.p['original_helpers']:
            closure.artifact(ROOT, entry)
        self.assertNotEqual(ref(RUNNER)['sha256'], self.p['original_helpers'][0]['sha256'])

    def test_additive_parent_has_actual_native_receipt_and_original_frozen_protocol(self):
        entries = parent_evidence()
        self.assertEqual(len(entries), 3)
        for entry in entries:
            closure.artifact(ROOT, entry)
        self.assertEqual(tick.load(closure.artifact(ROOT, entries[0]))['id'], 'AdaRES-Y92')

def rebind(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y92
        assert git('rev-parse', OLD_BRANCH).decode().strip() == OLD
        assert subprocess.run(['git', 'show-ref', '--verify', '--quiet', BRANCH], cwd=ROOT).returncode == 1
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next((row for row in work['candidates'] if row['id'] == CID))
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        assert candidate['code_commit'] == OLD and (not candidate.get('additive_version_binding'))
        assert not candidate.get('native_smoke_result') and (not candidate.get('admission_receipt'))
        original = copy.deepcopy(candidate)
        others = copy.deepcopy([row for row in work['candidates'] if row['id'] != CID])
        validate_cpu(tick.load(closure.artifact(ROOT, candidate['CPU_result'])))
        qref, rref = (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        from prepare_y93_evidence_hidden_skip import files
        paths = [path.relative_to(ROOT).as_posix() for path in files()]
        assert len(paths) == 7 and (not git('ls-files', '--', *paths).strip())
        for path in paths:
            assert (ROOT / path).read_bytes() == git('show', OLD + ':' + path)
        base = ROOT / ('pipeline/checks/y93_additive_binding_' + token)
        output, index, guards = (base.with_suffix('.json'), base.with_suffix('.index'), base.with_suffix('.log'))
        proposal_path = base.with_name(base.name + '_proposal.json')
        assert not any((path.exists() for path in (output, index, guards, proposal_path)))
        main_index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
        if not main_index.is_absolute():
            main_index = ROOT / main_index
        index_sha = hashlib.sha256(main_index.read_bytes()).hexdigest()
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        git('read-tree', Y92, env=env)
        for path in paths:
            blob = git('hash-object', '-w', '--stdin', data=(ROOT / path).read_bytes()).decode().strip()
            git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + path, env=env)
        tree = git('write-tree', env=env).decode().strip()
        commit = git('commit-tree', tree, '-p', Y92, '-m', 'Retain identical CPU-tested Y93 files additively on verified Y92').decode().strip()
        delta = source_delta(Y92, commit, paths)
        git('update-ref', BRANCH, commit, '0' * 40)
        proposed = copy.deepcopy(candidate)
        proposed.update(code_commit=commit, retained_branch=BRANCH)
        original_helpers = [ref('scripts/run_y93_native.py'), ref('scripts/rebind_y91_on_y90.py'), ref('scripts/run_y91_bound_native.py'), ref('scripts/retain_y93_cpu_native.py')]
        plans, originals = ({}, [])
        original_inputs = [candidate[key] for key in ('source_manifest', 'CPU_result', 'CPU_program', 'native_preparation', 'prepared_native_body', 'observer_core')]
        original_inputs += [ref('pipeline/goal.json')] + [ref(path) for path in paths] + parent_evidence()
        for name, key in PLAN_KEYS.items():
            old_entry = candidate[key]
            old_plan = tick.load(closure.artifact(ROOT, old_entry))
            before, body = parse_program(old_plan['program'])
            assert validate_frozen(candidate, old_plan, before) == 7
            frozen = copy.deepcopy(before)
            frozen['code_commit'] = commit
            suffix = name.replace('-', '')
            program = base.with_name(base.name + '_' + suffix + '_program.py')
            ppath = base.with_name(base.name + '_' + suffix + '_plan.json')
            assert not program.exists() and (not ppath.exists())
            program.write_text('FROZEN=' + repr(frozen) + '\n' + body, encoding='utf-8', newline='\n')
            plan = copy.deepcopy(old_plan)
            plan.update(code_commit=commit, program=ref(program), retained_branch=BRANCH, working_HEAD_preserved=Y92, supersedes=old_entry, additive_parent=Y92, execution_helper=ref(RUNNER), planned_command='python -B scripts/run_y93_bound_native_v2.py --server ' + name + ' --token <CURRENT_TOKEN>', required_resource='Fresh exclusive idle/unreserved authorized GPU0, after exact matching ' + OWNERS[name] + ' fixed30 checkpoint verification/canonical archive. No GPU1 or power/data/runtime substitution.')
            assert validate_frozen(proposed, plan, frozen) == 7
            tick.save(ppath, plan)
            compare_programs(old_plan['program'], plan['program'], proposed, OLD)
            plans[name] = ref(ppath)
            originals.extend([old_entry, old_plan['program']])
        proposal = dict(complete=True, id=CID, token=token, candidate=proposed, original=original, paths=paths, plans=plans, original_inputs=original_inputs + originals, main_index=str(main_index), main_index_sha256=index_sha, queue=qref, running=rref, other_candidates=others, original_helpers=original_helpers)
        tick.save(proposal_path, proposal)
        BindingGuards.proposal = proposal
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BindingGuards))
        guards.write_text(stream.getvalue(), encoding='utf-8', newline='\n')
        assert result.wasSuccessful() and result.testsRun == 16 and (not result.skipped)
        evidence = [ref(__file__), ref(proposal_path), ref(guards), ref(index)] + original_inputs + originals
        evidence += [ref(RUNNER)] + list(plans.values()) + original_helpers
        evidence += [tick.load(ROOT / entry['path'])['program'] for entry in plans.values()]
        tick.save(output, dict(complete=True, id=CID, token=token, observed_at=datetime.now().astimezone().isoformat(), code_commit=commit, parent_commit=Y92, working_HEAD=Y92, retained_branch=BRANCH, original_commit=OLD, original_branch=OLD_BRANCH, original_version=original['source_manifest'], original_CPU_result=original['CPU_result'], original_CPU_program=original['CPU_program'], original_native_preparation=original['native_preparation'], original_plans={name: original[key] for name, key in PLAN_KEYS.items()}, native_plans=plans, matching_current_owners=OWNERS, exact_delta=delta, scientific_files_identical=True, native_program_bodies_identical=True, only_frozen_code_commit_metadata_changed=True, original_evidence_retained=True, main_index_unchanged=True, other_candidates_unchanged=True, queue_unchanged=True, native_executed=False, CPU_scientific_experiments_replayed=False, admitted=False, production_sync=False, scientific_revision=False, guard_tests_passed=16, evidence=evidence))
        candidate['original_pre_y92_binding'] = dict(code_commit=OLD, retained_branch=OLD_BRANCH, source_manifest=original['source_manifest'], native_preparation=original['native_preparation'], native_smoke_plan=original['native_smoke_plan'], native_smoke_plan_server1=original['native_smoke_plan_server1'], native_smoke_plan_server2_gpu0=original['native_smoke_plan_server2_gpu0'])
        candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(output), additive_version_binding=ref(output), native_smoke_plan=plans['server-2'], native_smoke_plan_server1=plans['server-1'], native_smoke_plan_server2_gpu0=plans['server-2'], blocker='Current Y92/server1 and Y91/server2 reserve both authorized GPU0 resources; identical CPU-tested Y93 files have a new tested additive binding on Y92.', unblock_condition='Matching Y92/server1 or Y91/server2 naturally fixed30 complete, exact checkpoint-verified/canonically archived, then matching GPU0 freshly physically idle and unreserved.', next_action='Do not replay completed CPU/version/binding tests. Run scripts/run_y93_bound_native_v2.py --server server-1|server-2 --token CURRENT only at HEADY92 after exact matching owner fixed30 checkpoint verification/archive and fresh exclusive idle GPU0. Actual native PASS, matching full original freeze, admission, safe source selection/sync and increasing steps remain required. Further HEAD/owner changes require a NEW tested binding. Preserve original GPU1 frozen-review holds and separate Y57 pause.')
        candidate['evidence'].extend([ref(output)] + evidence)
        work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'], candidate_id=CID, kind='implementation_test', result='passed', command='python -B scripts/rebind_y93_on_y92_v2.py --token ' + token, observation='Sixteen actual source/payload/runtime/negative/current-owner guards passed for identical seven CPU-tested files on Y92. Scientific frozen inputs/native bodies unchanged except commit metadata. No native task execution, CPU scientific replay or efficacy.', evidence=[ref(output), ref(guards)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        assert git('rev-parse', 'HEAD').decode().strip() == Y92
        assert hashlib.sha256(main_index.read_bytes()).hexdigest() == index_sha
        assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        print(json.dumps(dict(binding=ref(output), code_commit=commit, plans=plans, tests=16, production_HEAD=Y92, native_executed=False, admitted=False)))
    finally:
        writer.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    rebind(parser.parse_args().token)
