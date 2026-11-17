"""Retain identical tested Y90 files additively without changing production HEAD."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from retain_y90_cpu_native import git
from check_ef_execution import validate_frozen
from y90_contract import validate_cpu
from y89_deployment import Y89, contract
import closure

CID = 'AdaRES-Y90'
OLD = '9a64d22c41bb8527ca7ca1abba4340546506b1b1'
OLD_BRANCH = 'refs/heads/adares-y90-query-peer-affinity'
BRANCH = 'refs/heads/adares-y90-query-peer-affinity-on-y89'
RUNNER = 'scripts/run_y90_bound_native.py'
PLAN_KEYS = {'server-1': 'native_smoke_plan_server1', 'server-2': 'native_smoke_plan_server2_gpu0'}


def parse_program(entry):
    path = closure.artifact(ROOT, entry)
    source = path.read_text(encoding='utf-8')
    parsed = ast.parse(source, feature_version=(3, 8))
    return ast.literal_eval(parsed.body[0].value), source.partition('\n')[2]


def compare_programs(old_entry, new_entry, candidate, old_commit):
    before, before_body = parse_program(old_entry)
    after, after_body = parse_program(new_entry)
    assert before['code_commit'] == old_commit and after['code_commit'] == candidate['code_commit']
    restored = copy.deepcopy(after)
    restored['code_commit'] = old_commit
    assert restored == before and after_body == before_body
    return after


def source_delta(parent, commit, paths):
    assert git('rev-parse', commit + '^').decode().strip() == parent
    rows = [row.split('\t') for row in git(
        'diff', '--no-renames', '--name-status', parent, commit).decode().splitlines()]
    assert len(rows) == 7 and all(len(row) == 2 and row[0] == 'A' for row in rows)
    assert {row[1] for row in rows} == set(paths)
    for path in paths:
        assert (ROOT / path).read_bytes() == git('show', OLD + ':' + path) == git('show', commit + ':' + path)
    return rows


def runner_module(path):
    spec = importlib.util.spec_from_file_location('y90_bound_native_guarded', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BindingGuards(unittest.TestCase):
    proposal = None

    def setUp(self):
        self.p = self.proposal
        self.c = self.p['candidate']

    def test_exact_seven_previously_tested_additions(self):
        source_delta(Y89, self.c['code_commit'], self.p['paths'])

    def test_production_HEAD_and_unrelated_index_unchanged(self):
        self.assertEqual(git('rev-parse', 'HEAD').decode().strip(), Y89)
        self.assertEqual(hashlib.sha256(Path(self.p['main_index']).read_bytes()).hexdigest(),
                         self.p['main_index_sha256'])

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

    def test_current_original_Y88_Y89_sources_preserved(self):
        self.assertIn(Y89, contract('cb4a5813755d38b3d1593bb803726aabd7d0a79b',
                                   Y89, 'AdaRES-Y88'))
        self.assertIn(Y89, contract(Y89, Y89, 'AdaRES-Y89'))

    def test_current_whitelists_and_pause_unchanged(self):
        servers = monitor.load_servers(str(ROOT))
        self.assertEqual(servers['server-1']['allowed_gpus'], [0])
        self.assertEqual(servers['server-2']['allowed_gpus'], [0])
        self.assertEqual(servers['server-6']['allowed_gpus'], [])
        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'],
                         '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')

    def test_native_runtime_alternatives_and_power(self):
        module = runner_module(ROOT / RUNNER)
        servers = monitor.load_servers(str(ROOT))
        for name in PLAN_KEYS:
            plan = tick.load(ROOT / self.p['plans'][name]['path'])
            frozen, _ = parse_program(plan['program'])
            self.assertEqual(module.runtime(name, self.c, frozen, servers[name]),
                             350. if name == 'server-1' else 280.)
            for key, value in [('runtime_physical_gpu', 1), ('runtime_power_limit_watts', 390.)]:
                with self.assertRaises(AssertionError):
                    module.runtime(name, self.c, dict(frozen, **{key: value}), servers[name])

    def test_runner_uses_current_owner_and_requires_exact_completion(self):
        module = runner_module(ROOT / RUNNER)
        self.assertEqual(module.OWNERS, {'server-1': 'AdaRES-Y88', 'server-2': 'AdaRES-Y89'})
        for name, owner in module.OWNERS.items():
            proof = dict(id=owner, complete=True, process_absent=True, CUDA_initialized=False,
                completed_epochs={phase: list(range(1, 31)) for phase in ('Train', 'Val')},
                checkpoints=[dict(epoch=epoch, scheduler_last_epoch=epoch, optimizer_state_count=1,
                    model_finite=True, optimizer_finite=True) for epoch in (30, 21)])
            module.validate_terminal(proof, name)
            for key, value in [('id', 'AdaRES-Y87'), ('process_absent', False), ('CUDA_initialized', True)]:
                with self.assertRaises(AssertionError):
                    module.validate_terminal(dict(proof, **{key: value}), name)
            proof['completed_epochs']['Train'].remove(12)
            with self.assertRaises(AssertionError):
                module.validate_terminal(proof, name)

    def test_original_plan_resource_and_path_mappings_retained(self):
        for name in PLAN_KEYS:
            old = tick.load(closure.artifact(ROOT, self.p['original'][PLAN_KEYS[name]]))
            new = tick.load(ROOT / self.p['plans'][name]['path'])
            for key in ('id', 'config', 'goal_sha256', 'server', 'gpu',
                        'runtime_power_limit_watts', 'resource_authorization'):
                self.assertEqual(new[key], old[key])
            self.assertFalse(new['native_executed'])
            self.assertFalse(new['admitted'])

    def test_runner_checks_new_binding_before_native_execution(self):
        source = (ROOT / RUNNER).read_text(encoding='utf-8')
        self.assertLess(source.index("candidate['additive_version_binding']"),
                        source.index('observation = live(server, 0)'))
        self.assertIn("assert binding['parent_commit'] == " + repr(Y89), source)
        self.assertIn("assert binding['code_commit'] == candidate['code_commit']", source)
        ast.parse(source, feature_version=(3, 8))

    def test_all_old_scientific_inputs_and_other_candidates_unchanged(self):
        for entry in self.p['original_inputs']:
            closure.artifact(ROOT, entry)
        current = tick.load(ROOT / 'pipeline/research_work.json')
        actual = [row for row in current['candidates'] if row['id'] != CID]
        self.assertEqual(actual, self.p['other_candidates'])
        self.assertEqual(self.p['queue'], ref('pipeline/queue.json'))
        self.assertEqual(self.p['running'], ref('pipeline/running.json'))


def create_runner():
    original = ROOT / 'scripts/run_y90_native.py'
    destination = ROOT / RUNNER
    assert not destination.exists()
    source = original.read_text(encoding='utf-8')
    owners = "OWNERS = {'server-1': 'AdaRES-Y88', 'server-2': 'AdaRES-Y87'}"
    assert source.count(owners) == 1
    source = source.replace(owners, "OWNERS = {'server-1': 'AdaRES-Y88', 'server-2': 'AdaRES-Y89'}")
    line = "    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y90')"
    guard = """
    binding = tick.load(closure.artifact(ROOT, candidate['additive_version_binding']))
    assert binding['complete'] is True and binding['id'] == candidate['id']
    assert binding['code_commit'] == candidate['code_commit']
    assert binding['parent_commit'] == """ + repr(Y89) + """
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == binding['working_HEAD']
    for dependency in binding['evidence']:
        closure.artifact(ROOT, dependency)
"""
    assert source.count(line) == 2
    source = source.replace(line, line + guard, 1)
    source = source.replace('python -B scripts/run_y90_native.py', 'python -B scripts/run_y90_bound_native.py')
    ast.parse(source, feature_version=(3, 8))
    destination.write_text(source, encoding='utf-8', newline='\n')
    return ref(original), ref(destination)


def rebind(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y89
        assert git('rev-parse', OLD_BRANCH).decode().strip() == OLD
        assert __import__('subprocess').run(['git', 'show-ref', '--verify', '--quiet', BRANCH],
                                          cwd=ROOT).returncode == 1
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(row for row in work['candidates'] if row['id'] == CID)
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        assert candidate['code_commit'] == OLD and not candidate.get('additive_version_binding')
        assert not candidate.get('native_smoke_result') and not candidate.get('admission_receipt')
        original = copy.deepcopy(candidate)
        other_candidates = copy.deepcopy([row for row in work['candidates'] if row['id'] != CID])
        validate_cpu(tick.load(closure.artifact(ROOT, candidate['CPU_result'])))
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        from prepare_y90_query_peer_affinity import files
        paths = [path.relative_to(ROOT).as_posix() for path in files()]
        assert len(paths) == 7 and not git('ls-files', '--', *paths).strip()
        for path in paths:
            assert (ROOT / path).read_bytes() == git('show', OLD + ':' + path)
        base = ROOT / ('pipeline/checks/y90_additive_binding_' + token)
        output, index, guards = base.with_suffix('.json'), base.with_suffix('.index'), base.with_suffix('.log')
        proposal_path = base.with_name(base.name + '_proposal.json')
        assert not any(path.exists() for path in (output, index, guards, proposal_path))
        main_index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
        if not main_index.is_absolute():
            main_index = ROOT / main_index
        index_sha = hashlib.sha256(main_index.read_bytes()).hexdigest()
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        git('read-tree', Y89, env=env)
        for path in paths:
            blob = git('hash-object', '-w', '--stdin', data=(ROOT / path).read_bytes()).decode().strip()
            git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + path, env=env)
        tree = git('write-tree', env=env).decode().strip()
        commit = git('commit-tree', tree, '-p', Y89,
                     '-m', 'Retain identical CPU-tested Y90 files additively on verified Y89').decode().strip()
        delta = source_delta(Y89, commit, paths)
        git('update-ref', BRANCH, commit, '0' * 40)
        proposed = copy.deepcopy(candidate)
        proposed.update(code_commit=commit, retained_branch=BRANCH)
        runner_refs = create_runner()
        plans, originals = {}, []
        original_inputs = [candidate[key] for key in
            ('source_manifest', 'CPU_result', 'CPU_program', 'native_preparation',
             'prepared_native_body', 'observer_core')]
        original_inputs += [ref('pipeline/goal.json')] + [ref(path) for path in paths]
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
            assert not program.exists() and not ppath.exists()
            program.write_text('FROZEN=' + repr(frozen) + '\n' + body, encoding='utf-8', newline='\n')
            plan = copy.deepcopy(old_plan)
            plan.update(code_commit=commit, program=ref(program), retained_branch=BRANCH,
                working_HEAD_preserved=Y89, supersedes=old_entry,
                additive_parent=Y89, execution_helper=runner_refs[1],
                planned_command='python -B scripts/run_y90_bound_native.py --server ' + name + ' --token <CURRENT_TOKEN>',
                required_resource='Fresh exclusive idle/unreserved authorized GPU0, after exact original '
                    + ('Y88/server1' if name == 'server-1' else 'Y89/server2')
                    + ' fixed30 checkpoint verification/canonical archive. No GPU1 or power/data/runtime substitution.')
            assert validate_frozen(proposed, plan, frozen) == 7
            tick.save(ppath, plan)
            compare_programs(old_plan['program'], plan['program'], proposed, OLD)
            plans[name] = ref(ppath)
            originals.extend([old_entry, old_plan['program']])
        proposal = dict(complete=True, id=CID, token=token, candidate=proposed, original=original,
            paths=paths, plans=plans, original_inputs=original_inputs + originals,
            main_index=str(main_index), main_index_sha256=index_sha,
            queue=qref, running=rref, other_candidates=other_candidates)
        tick.save(proposal_path, proposal)
        BindingGuards.proposal = proposal
        stream = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(BindingGuards)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        guards.write_text(stream.getvalue(), encoding='utf-8', newline='\n')
        assert result.wasSuccessful() and result.testsRun == 12 and not result.skipped
        evidence = [ref(__file__), ref(proposal_path), ref(guards), ref(index)] + original_inputs + originals
        evidence += list(runner_refs) + list(plans.values())
        for entry in plans.values():
            evidence.append(tick.load(ROOT / entry['path'])['program'])
        binding = dict(complete=True, id=CID, token=token, observed_at=datetime.now().astimezone().isoformat(),
            code_commit=commit, parent_commit=Y89, working_HEAD=Y89, retained_branch=BRANCH,
            original_commit=OLD, original_branch=OLD_BRANCH, original_version=original['source_manifest'],
            original_CPU_result=original['CPU_result'], original_CPU_program=original['CPU_program'],
            original_native_preparation=original['native_preparation'], original_plans={name: original[key]
                for name, key in PLAN_KEYS.items()}, native_plans=plans, exact_delta=delta,
            scientific_files_identical=True, native_program_bodies_identical=True,
            only_frozen_code_commit_metadata_changed=True, original_evidence_retained=True,
            main_index_unchanged=True, other_candidates_unchanged=True, queue_unchanged=True,
            native_executed=False, CPU_scientific_experiments_replayed=False, admitted=False,
            production_sync=False, scientific_revision=False, guard_tests_passed=12, evidence=evidence)
        tick.save(output, binding)
        candidate['original_pre_y89_binding'] = dict(code_commit=OLD, retained_branch=OLD_BRANCH,
            source_manifest=original['source_manifest'], native_preparation=original['native_preparation'],
            native_smoke_plan=original['native_smoke_plan'],
            native_smoke_plan_server1=original['native_smoke_plan_server1'],
            native_smoke_plan_server2_gpu0=original['native_smoke_plan_server2_gpu0'])
        candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(output),
            additive_version_binding=ref(output), native_smoke_plan=plans['server-2'],
            native_smoke_plan_server1=plans['server-1'], native_smoke_plan_server2_gpu0=plans['server-2'],
            blocker='Original Y88/server1 and Y89/server2 reserve both authorized GPU0 resources; '
                'identical tested Y90 bytes now have an explicit additive version binding on production Y89.',
            unblock_condition='Matching original Y88/server1 or Y89/server2 naturally fixed30 complete and '
                'exact checkpoint-verified/canonically archived; matching GPU0 physically idle and unreserved.',
            next_action='Do not replay CPU scientific tests, version retention or binding. Use '
                'scripts/run_y90_bound_native.py --server server-1|server-2 --token CURRENT only after '
                'matching exact completed owner and fresh exclusive idle GPU0. Actual native PASS required '
                'before new full frozen review/receipt/admission, safe source selection/sync/dispatch and '
                'increasing counters. Preserve current Y88/Y89, Y91 preparation and GPU1/Y57 pauses.')
        candidate['evidence'].extend([ref(output)] + evidence)
        work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id=CID, kind='implementation_test', result='passed',
            command='python -B scripts/rebind_y90_on_y89.py --token ' + token,
            observation='Twelve actual source/payload/runtime/negative guards passed for identical seven '
                'CPU-tested files on additive Y89 parent; all scientific frozen inputs/native bodies '
                'unchanged except commit metadata. No native execution, replayed CPU experiment or efficacy.',
            evidence=[ref(output), ref(guards)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        assert git('rev-parse', 'HEAD').decode().strip() == Y89
        assert hashlib.sha256(main_index.read_bytes()).hexdigest() == index_sha
        assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        print(json.dumps(dict(binding=ref(output), code_commit=commit, plans=plans, tests=12,
                              production_HEAD=Y89, native_executed=False, admitted=False)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    rebind(parser.parse_args().token)
