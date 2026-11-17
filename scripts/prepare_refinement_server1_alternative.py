"""Inactive server-1 plans for the unlaunched Y65/Y66 refinement experiments."""
import argparse
import ast
import copy
from datetime import datetime
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from prepare_y64_server1_alternative import ADAPT, adapted_body, idle, observe
import closure

IDS = ('AdaRES-Y65', 'AdaRES-Y66')


def assert_slot_unreserved(queue, running):
    assert not any(t['status'] in ('running', 'claiming')
                   for t in queue['gpu_queues']['server-1']), 'Canonical terminal archive must precede activation'
    assert not any(running.get('server-1', {}).values()), 'GPU reservation remains registered'


def activate(token, server, work, candidate):
    plan_ref = candidate['native_smoke_plan_server1']
    plan = tick.load(closure.artifact(ROOT, plan_ref))
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    assert frozen['runtime_server'] == 'server-1' and frozen['runtime_power_limit_watts'] == 350.0
    assert frozen['runtime_path_mappings'] == server['runtime_path_mappings']
    closure.artifact(ROOT, frozen['resource_authorization'])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == candidate['code_commit']
    assert_slot_unreserved(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)))
    observation = observe(server)
    assert idle(observation), 'GPU0 remains busy; preserve existing/foreign tasks'
    out = ROOT / ('pipeline/checks/' + candidate['id'].lower().replace('-', '_') + '_server1_activation_' + token + '.json')
    assert not out.exists()
    tick.save(out, dict(complete=True, token=token, id=candidate['id'], observed_at=datetime.now().astimezone().isoformat(),
        observation=observation, previous_plan=candidate['native_smoke_plan'], active_plan=plan_ref,
        no_queue_or_running_reservations=True, exclusive_idle_observed=True, native_executed=False,
        GPU_lease_and_idle_must_be_rechecked_by_runner=True, hardware_changed=False))
    candidate.update(stage='validation', required_server='server-1', native_smoke_plan=plan_ref)
    for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition', 'occupation_evidence'):
        candidate.pop(key, None)
    candidate['evidence'].append(ref(out))
    candidate['next_action'] = plan['planned_command'] + '; actual native PASS and frozen full evaluation protocol required before admission.'
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(activation=ref(out), active_plan=plan_ref, native_executed=False)))


def prepare(token, server, work):
    candidates = [next(c for c in work['candidates'] if c['id'] == cid) for cid in IDS]
    for c in candidates:
        assert c['stage'] == 'blocked' and c['blocker_kind'] == 'exclusive_gpu_slot'
        assert c['required_server'] == 'server-2' and not c.get('admission_receipt')
        assert not c.get('native_smoke_plan_server1')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    assert head == '882d61537fe7ffcb383c8f51f12fea36d1e53a6d'
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    live = next(t for t in monitor.load_queue(str(ROOT))['gpu_queues']['server-1'] if t['id'] == 'AdaRES-Y64')
    assert live['status'] == 'running' and live['launch_verified']
    assert live['code_commit'] == '9706a8c36bead4459668ab66a3d1abc39344ef8b'
    closure.admission(ROOT, live)
    observation = observe(server)
    actual_idle = idle(observation)
    grant = ref('pipeline/authorizations/server1_gpu0_20261025.json')
    closure.artifact(ROOT, grant)
    data_verified = ref('pipeline/checks/server1_data_verified_63b27dd8efe04a0bab6dc0291cefdbf0.json')
    assert data_verified['sha256'] == '8c590fa828fbc6c0df840c21ba88b1802cc0d2323eb0a124a57aa328b0c2361f'
    test_log = ROOT / ('pipeline/checks/refinement_server1_guards_' + token + '.log')
    assert not test_log.exists()
    test = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_refinement_server1_alternative.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    test_log.write_bytes(test.stdout + test.stderr)
    assert test.returncode == 0 and b'Ran 9 tests' in test_log.read_bytes()
    summary = []
    for candidate in candidates:
        original_ref = candidate['native_smoke_plan']
        plan = tick.load(closure.artifact(ROOT, original_ref))
        source = closure.artifact(ROOT, plan['program']).read_text(encoding='utf-8')
        frozen = ast.literal_eval(ast.parse(source).body[0].value)
        assert validate_frozen(candidate, plan, frozen) == 7 and candidate['code_commit'] == head
        assert plan['native_executed'] is False
        old_frozen = copy.deepcopy(frozen)
        frozen.update(runtime_server='server-1', runtime_power_limit_watts=350.0,
            runtime_path_mappings=copy.deepcopy(server['runtime_path_mappings']), resource_authorization=grant)
        assert all(frozen[k] == v for k, v in old_frozen.items())
        original_body = source.partition('\n')[2]
        body = adapted_body(original_body, 350.0)
        restored = body.replace(ADAPT, '').replace("== 'dl02-MS-7B73'", "== 'dl01-MS-7C60'")
        restored = restored.replace("== FROZEN['runtime_power_limit_watts']", '== 280.0')
        assert restored == original_body, 'Scientific diagnostic body must remain exact'
        short = candidate['id'].split('-')[1].lower()
        base = ROOT / ('pipeline/checks/' + short + '_server1_alternative_' + token)
        target = base.with_name(base.name + '_program.py')
        plan_path = base.with_name(base.name + '_plan.json')
        binding = base.with_name(base.name + '_binding.json')
        assert not any(p.exists() for p in (target, plan_path, binding))
        program = 'FROZEN=' + repr(frozen) + '\n' + body
        ast.parse(program, feature_version=(3, 8))
        target.write_text(program, encoding='utf-8', newline='\n')
        new_plan = copy.deepcopy(plan)
        new_plan.update(program=ref(target), server='server-1', gpu=0, previous_plan=original_ref,
            runtime_power_limit_watts=350.0, resource_authorization=grant,
            required_resource='Fresh exclusive idle authorized server-1 GPU0; no canonical queue/running '
                'reservation or compute PID; exact hostname, verified project private runtime and '
                'deterministic original data/path mappings; default350W unchanged. Preserve Y64 and Y57.',
            planned_command='python -B scripts/run_refinement_server1_native_probe.py ' + candidate['id'] + ' --token <CURRENT_TOKEN>',
            activation_required=True, native_executed=False)
        tick.save(plan_path, new_plan)
        assert validate_frozen(candidate, new_plan, frozen) == 7
        for field in ('goal', 'source'):
            bad = copy.deepcopy(frozen)
            if field == 'goal': bad['goal_sha256'] = '0' * 64
            else: bad['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
            refused = False
            try: validate_frozen(candidate, new_plan, bad)
            except AssertionError: refused = True
            assert refused
        tick.save(binding, dict(complete=True, token=token, id=candidate['id'],
            observed_at=datetime.now().astimezone().isoformat(), previous_plan=original_ref,
            inactive_server1_plan=ref(plan_path), actual_server1_observation=observation,
            actual_idle=actual_idle, preparation_only=True, activated=False,
            original_variant_and_YAML_identical=True, original_diagnostic_body_reversible_exact=True,
            goal_and_code_commit_unchanged=True, CPU_guard_tests_passed=9, test_log=ref(test_log),
            exact_original_data_verification=data_verified, resource_authorization=grant,
            source_and_path_changes='Only explicit host/power guards and deterministic configured path adaptation.',
            stale_goal_control_refused=True, tampered_source_control_refused=True,
            native_executed=False, hardware_changed=False, production_sync=False,
            queue_changed=False, running_changed=False, Y64_admission_preserved=True))
        candidate['native_smoke_plan_server1'] = ref(plan_path)
        additions = [ref(target), ref(plan_path), ref(binding), ref(test_log), ref(__file__),
            ref('scripts/run_refinement_server1_native_probe.py'), ref('scripts/test_refinement_server1_alternative.py')]
        candidate['evidence'].extend(additions)
        candidate['next_action'] = ('Use original current HEAD882d plan on genuinely free server-2 GPU0. '
            'If server-1 GPU0 frees first, verify canonical Y64 full terminal archive and exact live idle '
            'proof, then run prepare_refinement_server1_alternative.py activate --id ' + candidate['id'] +
            ' --token <CURRENT_TOKEN> and its native two-step runner. No current activation or GPU test. '
            'Actual native PASS and frozen full post-training protocol required for admission/dispatch.')
        summary.append(dict(id=candidate['id'], binding=ref(binding), inactive_plan=ref(plan_path)))
    closure.admission(ROOT, live)
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == head
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=IDS[0], kind='implementation_test', result='passed',
        command='python -B scripts/prepare_refinement_server1_alternative.py prepare --token ' + token,
        observation='Inactive server-1 Y65/Y66 plans prepared with byte-identical model/YAML and '
            'reversible exact scientific body;9 resource/path/activation guards and stale-goal/source '
            'controls passed. Y64 preserved; no GPU test, activation, admission or sync.',
        evidence=[ref(summary[0]['binding']['path']), ref(test_log)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(prepared=summary, tests=9, actual_idle=actual_idle,
        activated=False, native_executed=False, Y64_admission_preserved=True)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'activate'))
    parser.add_argument('--id', choices=IDS)
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 240)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-1']
        closure.validate_adares_server(ROOT, server)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        if args.operation == 'prepare':
            assert args.id is None
            prepare(args.token, server, work)
        else:
            assert args.id is not None
            candidate = next(c for c in work['candidates'] if c['id'] == args.id)
            assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
            assert candidate['required_server'] == 'server-2' and not candidate.get('admission_receipt')
            activate(args.token, server, work, candidate)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
