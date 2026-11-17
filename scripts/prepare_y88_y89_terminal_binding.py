"""Preserve old terminal plans and bind a new copy to the tested Y89 additions."""
import argparse
import ast
import copy
from datetime import datetime
import importlib
import json
from pathlib import Path
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y89_deployment import Y89, git
import closure


def generate():
    source = ROOT / 'scripts/finalize_y88_fixed30.py'
    original_tests = ROOT / 'scripts/test_y88_terminal_handoff.py'
    destination = ROOT / 'scripts/finalize_y88_fixed30_y89.py'
    test_destination = ROOT / 'scripts/test_y88_terminal_handoff_y89.py'
    assert not destination.exists() and not test_destination.exists()
    text = source.read_text(encoding='utf-8')
    assert text.count('from y88_terminal_contract import') == 1
    assert text.count('test_y88_terminal_handoff.py') == 2
    assert text.count('scripts/finalize_y88_fixed30.py') == 1
    text = text.replace('y88_terminal_contract', 'y88_y89_terminal_contract')
    text = text.replace('test_y88_terminal_handoff.py', 'test_y88_terminal_handoff_y89.py')
    text = text.replace('scripts/finalize_y88_fixed30.py', 'scripts/finalize_y88_fixed30_y89.py')
    tests = original_tests.read_text(encoding='utf-8')
    assert tests.count('import finalize_y88_fixed30 as archive') == 1
    assert tests.count('from y88_terminal_contract import') == 2
    tests = tests.replace('import finalize_y88_fixed30 as archive', 'import finalize_y88_fixed30_y89 as archive')
    tests = tests.replace('from y88_terminal_contract import', 'from y88_y89_terminal_contract import')
    for path, payload in ((destination, text), (test_destination, tests)):
        ast.parse(payload, feature_version=(3, 8))
        path.write_text(payload, encoding='utf-8', newline='\n')
    return destination, test_destination


def prepare(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y89
        original_refs = [ref(path) for path in ('scripts/finalize_y88_fixed30.py',
            'scripts/test_y88_terminal_handoff.py', 'scripts/y88_terminal_contract.py',
            'scripts/current_terminal_contract.py', 'scripts/prepare_y88_terminal_handoff.py')]
        generated = generate()
        archive = importlib.import_module('finalize_y88_fixed30_y89')
        server, queue, task, work, candidate = archive.current()
        before_queue, before_running = copy.deepcopy(queue), monitor.load_running(str(ROOT))
        oldref = copy.deepcopy(candidate['terminal_handoff_plan'])
        assert oldref['path'] == 'pipeline/checks/y88_terminal_preparation_75d4b121caff4ecab72bfeac9601eefe.json'
        assert oldref['sha256'] == '0de6507d9c7ee119823678ff3714680aa5d088f9e0a6cb43927a4f5bd8f4188f'
        old = tick.load(closure.artifact(ROOT, oldref))
        assert old['complete'] and old['id'] == 'AdaRES-Y88' and old['CPU_guard_tests_passed'] == 25
        for dependency in old['dependencies']:
            closure.artifact(ROOT, dependency)
        payload = copy.deepcopy(old['payload'])
        payload.update(archive.build_contract(task, server))
        restored = copy.deepcopy(payload)
        for key in ('allowed_markers', 'deployments'):
            restored[key] = old['payload'][key]
        assert restored == old['payload']
        observation_ref = candidate['latest_training_observation']
        observation = tick.load(closure.artifact(ROOT, observation_ref))
        assert observation['id'] == 'AdaRES-Y88' and observation['native_progress_confirmed'] is True
        age = (datetime.now().astimezone() - datetime.fromisoformat(observation['observed_local'])).total_seconds()
        assert 0 <= age < 300 and observation['source_marker'] in payload['allowed_markers']
        archive.validate_location(tick.load(closure.artifact(ROOT, old['declared_location'])), payload)
        base = ROOT / ('pipeline/checks/y88_y89_terminal_binding_' + token)
        output = base.with_suffix('.json')
        assert not output.exists()
        test_refs = []
        for pattern, count, label in [('test_y88_terminal_handoff_y89.py', 25, 'lifecycle'),
                                     ('test_y88_y89_binding.py', 6, 'additive')]:
            log = base.with_name(base.name + '_' + label + '.log')
            assert not log.exists()
            result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                '-s', 'scripts', '-p', pattern, '-v'], cwd=ROOT, capture_output=True, timeout=60)
            log.write_bytes(result.stdout + result.stderr)
            assert result.returncode == 0 and ('Ran ' + str(count) + ' tests').encode() in log.read_bytes()
            assert b'skipped' not in log.read_bytes()
            test_refs.append(ref(log))
        assert original_refs == [ref(item['path']) for item in original_refs]
        dependencies = old['dependencies'] + [ref(path) for path in generated] + [
            ref('scripts/y88_y89_terminal_contract.py'), ref('scripts/test_y88_y89_binding.py'),
            ref('scripts/y89_deployment.py'), ref(__file__)]
        plan = copy.deepcopy(old)
        plan.update(token=token, observed_at=datetime.now().astimezone().isoformat(),
            payload=payload, supersedes=oldref, original_run_and_checkpoint_unchanged=True,
            only_tested_additive_source_binding=True, original_helpers_unchanged=original_refs,
            latest_original_training_observation=observation_ref, additive_guard_tests_passed=6,
            dependencies=dependencies, evidence=old['evidence'] + test_refs + original_refs + dependencies,
            next_command='python -B scripts/finalize_y88_fixed30_y89.py --token <fresh-owned-token>')
        assert all(plan[key] is False for key in ('full_terminal_verification_executed',
            'checkpoint_loaded', 'archived', 'training_stopped', 'queue_changed', 'remote_mutations'))
        tick.save(output, plan)
        candidate['terminal_handoff_plan_before_y89'] = oldref
        candidate['terminal_handoff_plan'] = ref(output)
        candidate['evidence'].extend([ref(output), observation_ref] + test_refs + dependencies[-4:])
        assert monitor.load_queue(str(ROOT)) == before_queue and monitor.load_running(str(ROOT)) == before_running
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(binding=ref(output), lifecycle_tests=25, additive_tests=6,
            original_plan_preserved=oldref, checkpoint_loaded=False, archived=False)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    prepare(parser.parse_args().token)
