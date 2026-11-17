"""Test pending Y92 server-1 operations without consuming GPU or changing sources."""
import argparse
from datetime import datetime
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y92_deployment import Y91, Y92, git
import closure


def prepare(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y91
        before = {path: ref(path) for path in (
            'pipeline/queue.json', 'pipeline/running.json', 'pipeline/goal.json', 'SERVERS.yaml')}
        index = git('ls-files', '--stage', '-z')
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y92')
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        assert candidate['code_commit'] == Y92 and not candidate.get('native_smoke_result')
        assert not candidate.get('admission_receipt') and not candidate.get('execution_preparation')
        original_plan = candidate['native_smoke_plan']
        original_server = candidate['required_server']
        generation = ref('pipeline/checks/y92_execution_helpers_' + token + '.json')
        generated = tick.load(closure.artifact(ROOT, generation))
        for evidence in generated['evidence'] + generated['originals']:
            closure.artifact(ROOT, evidence)
        output = ROOT / ('pipeline/checks/y92_execution_preparation_' + token + '.json')
        log = output.with_suffix('.log')
        assert not output.exists() and not log.exists()
        response = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
            '-s', 'scripts', '-p', 'test_y92_execution_additional.py', '-v'],
            cwd=ROOT, capture_output=True, timeout=90)
        log.write_bytes(response.stdout + response.stderr)
        assert response.returncode == 0 and b'Ran 17 tests' in log.read_bytes()
        assert b'skipped' not in log.read_bytes()
        helpers = [ref('scripts/' + name) for name in (
            'advance_y92.py', 'y92_deployment.py', 'test_y92_execution_guards.py',
            'test_y92_execution_additional.py', 'test_y92_admission.py', 'prepare_y92_execution.py')]
        assert all(ref(path) == value for path, value in before.items())
        assert git('ls-files', '--stage', '-z') == index
        tick.save(output, dict(complete=True, token=token, id=candidate['id'], code_commit=Y92,
            prepared_at=datetime.now().astimezone().isoformat(), source_resource_protocol_guards=17,
            scope=dict(server='server-1', gpu=0, power_limit_watts=350.),
            evidence=helpers + [ref(log), generation, candidate['additive_version_binding']],
            actual_native_executed=False, actual_native_admission_guards_executed=False,
            actual_native_admission_tests_require_real_result=True,
            admitted=False, queued=False, production_HEAD_preserved=Y91,
            index_queue_running_preserved=True, signed_originals_unchanged=True,
            scientific_files_and_native_bodies_unchanged=True, CPU_scientific_tests_replayed=False,
            current_native_plan_activation_unchanged=True,
            frozen_review_parent='AdaRES-Y90',
            full_original_review_training_in_memory_identity_tested=True,
            next_action='After exact original Y90 full30 checkpoint verification/archive and fresh exclusive '
                'idle server-1 GPU0, run retained Y92 bound native server1 plan. On actual native PASS, '
                'prepare a NEW explicitly tested Y91 terminal binding for additive Y92 using that real native '
                'evidence before selecting Y92; then select, admit and canonical dispatch/observe same tick. '
                'Full original Y90 server1/350W review/training protocol is inherited with method-ID only. '
                'Server2 original native alternative remains immutable and needs its own matching execution '
                'binding if selected. No GPU1/Y57 use, source selection, receipt or launch now.'))
        candidate['execution_preparation'] = ref(output)
        candidate['evidence'].extend(helpers + [ref(output), ref(log)])
        candidate['next_action'] = tick.load(output)['next_action']
        assert candidate['native_smoke_plan'] == original_plan and candidate['required_server'] == original_server
        work['tick_attempt'] = dict(
            tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id=candidate['id'], kind='implementation_test', result='passed',
            command='python -B scripts/prepare_y92_execution.py --token ' + token,
            observation='Seventeen actual pending-source/protocol/runtime/current-owner/activation guards '
                'passed without GPU work, CPU scientific replay or native-result fixtures. Actual admission '
                'suite correctly refuses missing native evidence. Both original owners/queue preserved.',
            evidence=[ref(output), ref(log)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(preparation=ref(output), operational_guards=17,
                              native_executed=False, admitted=False)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    prepare(parser.parse_args().token)
