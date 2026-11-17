"""Retain identical tested Y96 bytes on actual-native Y94 without selecting HEAD."""
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
from y95_deployment import Y94, Y95, contract
from check_ef_execution import validate_frozen
from y96_contract import validate_cpu
from rebind_y90_on_y89 import parse_program, compare_programs, runner_module
import closure
CID = 'AdaRES-Y96'
OLD = '994db457a3b00444b6ace4f0c9b7f3b6e047c83e'
OLD_BRANCH = 'refs/heads/adares-y96-seed-query-skip'
BRANCH = 'refs/heads/adares-y96-seed-query-skip-on-y95'
RUNNER = 'scripts/run_y96_bound_native_y95.py'
OWNERS = {'server-1': 'AdaRES-Y94', 'server-2': 'AdaRES-Y95'}
PLAN_KEYS = {'server-1': 'native_smoke_plan_server1', 'server-2': 'native_smoke_plan_server2_gpu0'}
BINDING = "\ndef validate_binding(binding, candidate, head):\n    assert binding['complete'] is True and binding['id'] == candidate['id'] == 'AdaRES-Y96'\n    assert binding['code_commit'] == candidate['code_commit']\n    assert binding['parent_commit'] == binding['working_HEAD'] == head == PARENT\n    assert binding['matching_current_owners'] == OWNERS\n    assert binding['scientific_files_identical'] is True\n    assert binding['native_program_bodies_identical'] is True\n    assert binding['only_frozen_code_commit_metadata_changed'] is True\n"

class Runner(ast.NodeTransformer):

    def visit_Constant(self, node):
        if node.value == 'daa93b4375f144f0e79da92f85fe1fe27e877e18':
            node.value = Y95
        elif isinstance(node.value, str):
            node.value = node.value.replace('run_y96_native.py', 'run_y96_bound_native_y95.py')
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any((isinstance(target, ast.Name) and target.id == 'OWNERS' for target in node.targets)):
            node.value = ast.parse(repr(OWNERS), mode='eval').body
        return node

    def visit_FunctionDef(self, node):
        if node.name == 'validate_binding':
            return ast.parse(BINDING).body[0]
        return self.generic_visit(node)

def parent_evidence():
    from advance_y95 import prepared, validate_result
    candidate, _, _, frozen = prepared(tick.load(ROOT / 'pipeline/research_work.json'))
    entry = candidate['native_smoke_result']
    validate_result(tick.load(closure.artifact(ROOT, entry)), candidate, frozen)
    return entry

class BindingGuards(unittest.TestCase):
    proposal = None

    def test_only_seven_identical_tested_additions(self):
        p = self.proposal
        commit = p['candidate']['code_commit']
        self.assertEqual(git('rev-parse', commit + '^').decode().strip(), Y95)
        rows = [row.split('\t') for row in git('diff', '--no-renames', '--name-status', Y95, commit).decode().splitlines()]
        self.assertEqual(rows, p['delta'])
        self.assertEqual(len(rows), 7)
        self.assertTrue(all((status == 'A' for status, _ in rows)))
        self.assertEqual({path for _, path in rows}, set(p['paths']))
        for path in p['paths']:
            self.assertEqual((ROOT / path).read_bytes(), git('show', OLD + ':' + path))
            self.assertEqual((ROOT / path).read_bytes(), git('show', commit + ':' + path))

    def test_original_branch_CPU_evidence_and_callbacks_preserved(self):
        p = self.proposal
        self.assertEqual(git('rev-parse', OLD_BRANCH).decode().strip(), OLD)
        validate_cpu(tick.load(closure.artifact(ROOT, p['original']['CPU_result'])))
        for key in ('CPU_result', 'CPU_program', 'native_preparation', 'prepared_native_body', 'observer_core'):
            self.assertEqual(p['candidate'][key], p['original'][key])
            closure.artifact(ROOT, p['original'][key])

    def test_both_native_programs_identical_except_commit_metadata(self):
        p = self.proposal
        for name, key in PLAN_KEYS.items():
            old = tick.load(closure.artifact(ROOT, p['original'][key]))
            new = tick.load(closure.artifact(ROOT, p['plans'][name]))
            frozen = compare_programs(old['program'], new['program'], p['candidate'], OLD)
            self.assertEqual(validate_frozen(p['candidate'], new, frozen), 7)
            self.assertEqual(new['execution_helper'], ref(RUNNER))
            ast.parse((ROOT / new['program']['path']).read_text(encoding='utf-8'), feature_version=(3, 8))

    def test_wrong_source_goal_or_commit_is_refused(self):
        p = self.proposal
        for entry in p['plans'].values():
            plan = tick.load(closure.artifact(ROOT, entry))
            frozen, _ = parse_program(plan['program'])
            for field in ('source', 'goal', 'commit'):
                changed = copy.deepcopy(frozen)
                if field == 'source':
                    changed['variant']['python_files']['dec.py']['base64'] = 'YmFk'
                elif field == 'goal':
                    changed['goal_sha256'] = '0' * 64
                else:
                    changed['code_commit'] = OLD
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    validate_frozen(p['candidate'], plan, changed)

    def test_exact_private_runtimes_and_negative_device_controls(self):
        p = self.proposal
        module = runner_module(ROOT / RUNNER)
        for name, entry in p['plans'].items():
            plan = tick.load(closure.artifact(ROOT, entry))
            frozen, _ = parse_program(plan['program'])
            server = monitor.load_servers(str(ROOT))[name]
            self.assertEqual(module.runtime(name, p['candidate'], frozen, server), 350.0 if name == 'server-1' else 280.0)
            for key, value in (('runtime_physical_gpu', 1), ('runtime_server', 'server-6'), ('runtime_power_limit_watts', 390.0)):
                with self.assertRaises(AssertionError):
                    module.runtime(name, p['candidate'], dict(frozen, **{key: value}), server)
            for key, value in (('expected_hostname', 'foreign'), ('python', '/wrong/python'), ('allowed_gpus', [0, 1])):
                with self.assertRaises(AssertionError):
                    module.runtime(name, p['candidate'], frozen, dict(server, **{key: value}))

    def test_only_matching_current_owner_full_checkpoint_schema(self):
        module = runner_module(ROOT / RUNNER)
        self.assertEqual(module.OWNERS, OWNERS)
        for name, owner in OWNERS.items():
            proof = dict(id=owner, complete=True, process_absent=True, CUDA_initialized=False, completed_epochs={phase: list(range(1, 31)) for phase in ('Train', 'Val')}, checkpoints=[dict(epoch=epoch, scheduler_last_epoch=epoch, optimizer_state_count=1, model_finite=True, optimizer_finite=True) for epoch in (30, 21)])
            module.validate_terminal(proof, name)
            for key, value in (('id', 'AdaRES-Y92'), ('process_absent', False), ('CUDA_initialized', True)):
                with self.assertRaises(AssertionError):
                    module.validate_terminal(dict(proof, **{key: value}), name)
            for key, value in (('epoch', 29), ('scheduler_last_epoch', 0), ('optimizer_state_count', 0), ('model_finite', False), ('optimizer_finite', False)):
                changed = copy.deepcopy(proof)
                changed['checkpoints'][0][key] = value
                with self.assertRaises(AssertionError):
                    module.validate_terminal(changed, name)

    def test_actual_HEAD_and_complete_identical_binding_required(self):
        p = self.proposal
        module = runner_module(ROOT / RUNNER)
        binding = dict(complete=True, id=CID, code_commit=p['candidate']['code_commit'], parent_commit=Y95, working_HEAD=Y95, matching_current_owners=OWNERS, scientific_files_identical=True, native_program_bodies_identical=True, only_frozen_code_commit_metadata_changed=True)
        module.validate_binding(binding, p['candidate'], Y95)
        for key, value in (('id', 'AdaRES-Y95'), ('parent_commit', Y94), ('working_HEAD', Y94), ('scientific_files_identical', False), ('native_program_bodies_identical', False), ('only_frozen_code_commit_metadata_changed', False), ('matching_current_owners', {'server-1': 'AdaRES-Y92', 'server-2': 'AdaRES-Y94'})):
            with self.assertRaises(AssertionError):
                module.validate_binding(dict(binding, **{key: value}), p['candidate'], Y95)
        with self.assertRaises(AssertionError):
            module.validate_binding(binding, p['candidate'], Y94)

    def test_actual_parent_native_and_preserved_current_run_sources(self):
        self.assertEqual(parent_evidence(), self.proposal['parent_native'])
        self.assertEqual(list(contract(Y94, Y95, 'AdaRES-Y94')), [Y94, Y95])
        queue = monitor.load_queue(str(ROOT))
        for name, cid, pid, version in (('server-1', 'AdaRES-Y94', 218513, Y94), ('server-2', 'AdaRES-Y95', 34484, Y95)):
            task = next((row for row in queue['gpu_queues'][name] if row['id'] == cid))
            self.assertEqual((task['status'], task['pid'], task['code_commit']), ('running', pid, version))
            self.assertTrue(task['launch_verified'])
            closure.admission(ROOT, task)

    def test_production_HEAD_index_queue_and_other_candidates_preserved(self):
        p = self.proposal
        self.assertEqual(git('rev-parse', 'HEAD').decode().strip(), Y95)
        self.assertEqual(hashlib.sha256(Path(p['main_index']).read_bytes()).hexdigest(), p['main_index_sha256'])
        for path, entry in p['before'].items():
            self.assertEqual(ref(path), entry)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.assertEqual([row for row in work['candidates'] if row['id'] != CID], p['other_candidates'])
        for entry in p['original_helpers']:
            closure.artifact(ROOT, entry)

    def test_resource_pauses_and_checks_precede_remote_activation(self):
        servers = monitor.load_servers(str(ROOT))
        self.assertEqual(servers['server-1']['allowed_gpus'], [0])
        self.assertEqual(servers['server-2']['allowed_gpus'], [0])
        self.assertEqual(servers['server-6']['allowed_gpus'], [])
        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'], '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')
        source = (ROOT / RUNNER).read_text(encoding='utf-8')
        self.assertLess(source.index('validate_binding(binding, candidate, closure.git('), source.index('observation = live(server, 0)'))
        self.assertLess(source.index('validate_terminal(terminal, name)'), source.index('observation = live(server, 0)'))
        self.assertIn("assert plan['execution_helper'] == ref(__file__)", source)
        ast.parse(source, feature_version=(3, 8))

def rebind(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y95
        assert subprocess.run(['git', 'show-ref', '--verify', '--quiet', BRANCH], cwd=ROOT).returncode == 1
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next((row for row in work['candidates'] if row['id'] == CID))
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        assert candidate['code_commit'] == OLD and (not candidate.get('native_smoke_result'))
        assert not candidate.get('admission_receipt') and (not candidate.get('original_pre_y95_binding'))
        original = copy.deepcopy(candidate)
        from prepare_y96_seed_query_skip import files
        paths = [path.relative_to(ROOT).as_posix() for path in files()]
        assert len(paths) == 7 and (not git('ls-files', '--', *paths).strip())
        base = ROOT / ('pipeline/checks/y96_y95_additive_binding_' + token)
        output, index, log = (base.with_suffix('.json'), base.with_suffix('.index'), base.with_suffix('.log'))
        assert not any((path.exists() for path in (output, index, log, ROOT / RUNNER)))
        before = {path: ref(path) for path in ('pipeline/queue.json', 'pipeline/running.json', 'pipeline/goal.json', 'SERVERS.yaml')}
        main_index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
        if not main_index.is_absolute():
            main_index = ROOT / main_index
        index_sha = hashlib.sha256(main_index.read_bytes()).hexdigest()
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        git('read-tree', Y95, env=env)
        for path in paths:
            raw = (ROOT / path).read_bytes()
            assert raw == git('show', OLD + ':' + path)
            blob = git('hash-object', '-w', '--stdin', data=raw).decode().strip()
            git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + path, env=env)
        tree = git('write-tree', env=env).decode().strip()
        commit = git('commit-tree', tree, '-p', Y95, '-m', 'Retain identical CPU-tested Y96 additions on native-tested Y95').decode().strip()
        git('update-ref', BRANCH, commit, '0' * 40)
        source = ROOT / 'scripts/run_y96_native.py'
        parsed = Runner().visit(ast.parse(source.read_text(encoding='utf-8')))
        text = ast.unparse(ast.fix_missing_locations(parsed)) + '\n'
        ast.parse(text, feature_version=(3, 8))
        (ROOT / RUNNER).write_text(text, encoding='utf-8', newline='\n')
        proposed = copy.deepcopy(candidate)
        proposed.update(code_commit=commit, retained_branch=BRANCH)
        plans, old_inputs = ({}, [])
        for name, key in PLAN_KEYS.items():
            oldref = original[key]
            old = tick.load(closure.artifact(ROOT, oldref))
            frozen, body = parse_program(old['program'])
            assert validate_frozen(original, old, frozen) == 7
            frozen['code_commit'] = commit
            program = base.with_name(base.name + '_' + name.replace('-', '') + '_program.py')
            plan_path = base.with_name(base.name + '_' + name.replace('-', '') + '_plan.json')
            assert not program.exists() and (not plan_path.exists())
            program.write_text('FROZEN=' + repr(frozen) + '\n' + body, encoding='utf-8', newline='\n')
            plan = copy.deepcopy(old)
            plan.update(code_commit=commit, program=ref(program), retained_branch=BRANCH, supersedes=oldref, additive_parent=Y95, execution_helper=ref(RUNNER), planned_command='python -B ' + RUNNER + ' --server ' + name + ' --token <CURRENT>', working_HEAD_preserved=Y95, required_resource='At HEADY95 only after exact matching ' + OWNERS[name] + ' natural fixed30/full finite original checkpoint archive and fresh exclusive authorized idle/unreserved GPU0. No GPU1, power or runtime substitution.')
            tick.save(plan_path, plan)
            compare_programs(old['program'], plan['program'], proposed, OLD)
            plans[name] = ref(plan_path)
            old_inputs.extend([oldref, old['program']])
        delta = [row.split('\t') for row in git('diff', '--no-renames', '--name-status', Y95, commit).decode().splitlines()]
        proposal = dict(candidate=proposed, original=original, paths=paths, delta=delta, plans=plans, parent_native=parent_evidence(), main_index=str(main_index), main_index_sha256=index_sha, before=before, other_candidates=[row for row in work['candidates'] if row['id'] != CID], original_helpers=[ref(source), ref('scripts/retain_y96_cpu_native.py'), ref('scripts/test_y96_runtime_plan.py')])
        BindingGuards.proposal = proposal
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BindingGuards))
        log.write_text(stream.getvalue(), encoding='utf-8', newline='\n')
        assert result.wasSuccessful() and result.testsRun == 10 and (not result.skipped), str(log)
        dependencies = [ref(__file__), ref(RUNNER), ref(log), ref(index), proposal['parent_native']]
        dependencies += old_inputs + list(plans.values()) + proposal['original_helpers']
        dependencies += [original[key] for key in ('source_manifest', 'CPU_result', 'CPU_program', 'native_preparation', 'prepared_native_body', 'observer_core')]
        dependencies += [tick.load(closure.artifact(ROOT, entry))['program'] for entry in plans.values()]
        tick.save(output, dict(complete=True, id=CID, token=token, observed_at=datetime.now().astimezone().isoformat(), code_commit=commit, parent_commit=Y95, working_HEAD=Y95, prepared_from_HEAD=Y95, matching_current_owners=OWNERS, retained_branch=BRANCH, original_commit=OLD, scientific_files_identical=True, native_program_bodies_identical=True, only_frozen_code_commit_metadata_changed=True, guard_tests_passed=10, native_executed=False, admitted=False, scientific_revision=False, actual_parent_native=proposal['parent_native'], native_plans=plans, original_binding=original['source_manifest'], production_HEAD_index_queue_preserved=True, evidence=dependencies))
        candidate['original_pre_y95_binding'] = {key: original[key] for key in ('code_commit', 'retained_branch', 'source_manifest', 'native_smoke_plan', 'native_smoke_plan_server1', 'native_smoke_plan_server2_gpu0')}
        candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(output), additive_version_binding=ref(output), native_smoke_plan=plans['server-2'], native_smoke_plan_server1=plans['server-1'], native_smoke_plan_server2_gpu0=plans['server-2'], blocker='Current Y94/server1 and original Y95/server2 occupy both authorized GPU0 resources; identical Y96 bytes now have a tested HEADY95/current-owner execution binding.', unblock_condition='HEADY95, matching Y94/server1 or Y95/server2 natural30/full finite original checkpoint/archive, then fresh exclusive idle/unreserved authorized GPU0.', next_action='Do not replay CPU/version/binding tests. Only at HEADY95 use run_y96_bound_native_y95.py after matching Y94/server1 or Y95/server2 natural30/full finite original checkpoint verification and canonical archive plus fresh exclusive idle/unreserved GPU0. On actual native PASS complete original full freeze, receipt/admission/source selection/safe dispatch and increasing native counters in the same tick. Changed HEAD/owner requires a NEW tested binding. GPU1/Y57 stay paused; efficacy unknown.')
        candidate['evidence'].extend([ref(output)] + dependencies)
        work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'], candidate_id=CID, kind='implementation_test', result='passed', command='python -B scripts/rebind_y96_on_y95.py --token ' + token, observation='Ten new operational guards passed on identical actual CPU-tested seven files and unchanged native program bodies, exact Y94 parent native evidence and matching Y94/server1-Y95/server2 full-checkpoint resource conditions. No scientific CPU replay, native GPU execution, admission or launch.', evidence=[ref(output), ref(log)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(binding=ref(output), code_commit=commit, guards=10, native_executed=False, production_HEAD_preserved=Y95)))
    finally:
        writer.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    rebind(parser.parse_args().token)
