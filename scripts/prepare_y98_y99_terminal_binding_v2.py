"""Specialize unchanged Y98 lifecycle checks after actual Y99 native success."""
import argparse
import ast
import copy
from datetime import datetime
import importlib
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y99_deployment_v2 import Y98, Y99, git
from y98_y99_terminal_contract_v2 import native_evidence
import closure

class Specialize(ast.NodeTransformer):

    def __init__(self):
        self.expectation_updates = 0

    def visit_ImportFrom(self, node):
        if node.module == 'y98_terminal_contract':
            node.module = 'y98_y99_terminal_contract_v2'
        return self.generic_visit(node)

    def visit_alias(self, node):
        if node.name == 'finalize_y98_fixed30':
            node.name = 'finalize_y98_fixed30_y99_v2'
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for old, new in (('test_y98_terminal_handoff.py', 'test_y98_terminal_handoff_y99_v2.py'), ('scripts/finalize_y98_fixed30.py', 'scripts/finalize_y98_fixed30_y99_v2.py'), ('scripts/y98_terminal_contract.py', 'scripts/y98_y99_terminal_contract_v2.py'), ('pipeline/checks/y98_server2_fixed30_', 'pipeline/checks/y98_server2_fixed30_y99_')):
                node.value = node.value.replace(old, new)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'assertEqual' and (len(node.args) == 2) and isinstance(node.args[1], ast.List) and (len(node.args[1].elts) == 1) and isinstance(node.args[1].elts[0], ast.Name) and (node.args[1].elts[0].id == 'Y98'):
            assert isinstance(node.args[0], ast.Subscript)
            assert node.args[0].slice.value == 'allowed_markers'
            node.args[1].elts.append(ast.Name(id='Y99', ctx=ast.Load()))
            self.expectation_updates += 1
        return node

def prepare(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y98
        before = {path: ref(path) for path in ('pipeline/queue.json', 'pipeline/running.json', 'pipeline/goal.json', 'SERVERS.yaml', 'pipeline/admissions/AdaRES-Y98.json')}
        index = git('ls-files', '--stage', '-z')
        work = tick.load(ROOT / 'pipeline/research_work.json')
        original = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y98'))
        additive = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y99'))
        assert original['stage'] == 'training' and (not original.get('terminal_evidence'))
        assert additive['stage'] == 'validation' and (not additive.get('admission_receipt'))
        actual = native_evidence(work)
        assert not original.get('pre_y99_terminal_handoff_plan')
        oldref = original['terminal_handoff_plan']
        old = tick.load(closure.artifact(ROOT, oldref))
        originals = [ref('scripts/' + name) for name in ('finalize_y98_fixed30.py', 'test_y98_terminal_handoff.py', 'y98_terminal_contract.py')]
        generated = []
        for source_name, target_name, count in (('finalize_y98_fixed30.py', 'finalize_y98_fixed30_y99_v2.py', 0), ('test_y98_terminal_handoff.py', 'test_y98_terminal_handoff_y99_v2.py', 1)):
            path = ROOT / 'scripts' / target_name
            assert not path.exists()
            transform = Specialize()
            tree = transform.visit(ast.parse((ROOT / 'scripts' / source_name).read_text(encoding='utf-8')))
            assert transform.expectation_updates == count
            if count:
                tree.body.insert(1, ast.ImportFrom(module='y99_deployment_v2', names=[ast.alias(name='Y99')], level=0))
            source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
            ast.parse(source, feature_version=(3, 8))
            path.write_text(source, encoding='utf-8', newline='\n')
            generated.append(ref(path))
        archive = importlib.import_module('finalize_y98_fixed30_y99_v2')
        server, _, task, _, _ = archive.current()
        payload = copy.deepcopy(old['payload'])
        payload.update(archive.build_contract(task, server))
        assert set(payload) == set(old['payload'])
        assert all((payload[key] == old['payload'][key] for key in set(payload) - {'allowed_markers', 'deployments'}))
        assert payload['allowed_markers'] == [Y98, Y99]
        base = ROOT / ('pipeline/checks/y98_y99_terminal_binding_' + token)
        output = base.with_suffix('.json')
        assert not output.exists()
        logs = []
        for pattern, count, label in (('test_y98_terminal_handoff_y99_v2.py', 29, 'lifecycle'), ('test_y98_y99_terminal_binding_v2.py', 8, 'additive')):
            path = base.with_name(base.name + '_' + label + '.log')
            assert not path.exists()
            response = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts', '-p', pattern, '-v'], cwd=ROOT, capture_output=True, timeout=90)
            path.write_bytes(response.stdout + response.stderr)
            assert response.returncode == 0, str(path)
            assert ('Ran ' + str(count) + ' tests').encode() in path.read_bytes()
            assert b'skipped' not in path.read_bytes()
            logs.append(ref(path))
        dependencies = old['dependencies'] + generated + [ref(__file__)] + [ref('scripts/' + name) for name in ('y98_y99_terminal_contract_v2.py', 'test_y98_y99_terminal_binding_v2.py', 'y99_deployment_v2.py', 'advance_y99_v2.py', 'y99_contract.py', 'run_y99_bound_native_y98.py')] + [actual, additive['additive_version_binding'], additive['native_smoke_plan']]
        for entry in dependencies:
            closure.artifact(ROOT, entry)
        plan = copy.deepcopy(old)
        plan.update(token=token, observed_at=datetime.now().astimezone().isoformat(), payload=payload, dependencies=dependencies, supersedes=oldref, CPU_guard_tests_passed=29, additive_guard_tests_passed=8, tested_additive_native_evidence=actual, next_command='python -B scripts/finalize_y98_fixed30_y99_v2.py --token <fresh-owned-token>', original_helpers_unchanged=True, original_run_and_protocol_unchanged=True, full_terminal_verification_executed=False, checkpoint_loaded=False, archived=False, training_stopped=False, queue_changed=False, remote_mutations=False, evidence=old['evidence'] + dependencies + logs)
        tick.save(output, plan)
        assert all((ref(path) == entry for path, entry in before.items()))
        assert git('ls-files', '--stage', '-z') == index
        assert [ref(entry['path']) for entry in originals] == originals
        original['pre_y99_terminal_handoff_plan'] = oldref
        original['terminal_handoff_plan'] = ref(output)
        original['evidence'].extend([ref(output)] + generated + logs)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(binding=ref(output), lifecycle_guards=29, additive_guards=8)))
    finally:
        writer.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    prepare(parser.parse_args().token)
