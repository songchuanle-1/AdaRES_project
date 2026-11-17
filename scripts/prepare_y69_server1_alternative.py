"""Prepare an inactive server-1 Y69 plan; activate only after exact idle proof."""
import argparse
import ast
import copy
from datetime import datetime
import json
import shlex
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure

ADAPT = '''def adapt_path(value):
    assert P['path_mappings'] == FROZEN['runtime_path_mappings']
    for mapping in P['path_mappings']:
        old, new = mapping['source'].rstrip('/'), mapping['target'].rstrip('/')
        if value == old or value.startswith(old + '/'):
            value = new + value[len(old):]
    return value
cfg.train.pretrain = adapt_path(cfg.train.pretrain)
cfg.work_dir = adapt_path(cfg.work_dir)
for split in ('train', 'val'):
    cfg.data[split].data_root = adapt_path(cfg.data[split].data_root)
'''


def adapted_body(body, power):
    assert power == 350.0
    host = "== 'dl01-MS-7C60'"
    limit = "== FROZEN['runtime_power_limit_watts']"
    anchor = 'cfg = gorilla.Config.fromfile(str(config_path))\n'
    assert body.count(host) == body.count(limit) == body.count(anchor) == 1
    result = body.replace(host, "== 'dl02-MS-7B73'")
    result = result.replace(anchor, anchor + ADAPT)
    assert result.replace(ADAPT, '').replace("== 'dl02-MS-7B73'", host) == body
    ast.parse(result, feature_version=(3, 8))
    return result


REMOTE = '''import json,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu,power.limit','--format=csv,noheader,nounits']).decode().strip()
compute=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
print(json.dumps(dict(hostname='dl02-MS-7B73',gpu=gpu,compute=compute)))
'''


def idle(observation):
    from math import isfinite
    assert observation['hostname'] == 'dl02-MS-7B73'
    used, util, power = [float(v.strip()) for v in observation['gpu'].split(',')]
    assert all(isfinite(v) for v in (used, util, power))
    assert power == 350.0
    assert used >= 0 and 0 <= util <= 100
    return used < 1024 and util < 10 and not observation['compute'].strip()


def observe(server):
    command = shlex.quote(server['python']) + ' -B -c ' + shlex.quote(REMOTE)
    return json.loads(monitor.ssh(server['host'], server['user'], command, timeout=40))


def prepare(token, server, work, candidate):
    assert not candidate.get('native_smoke_plan_server1')
    assert candidate['code_commit'] == '9acd73cdc1b96be003c8450a9e22057248a0f531'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip() == candidate['code_commit']
    original_ref = candidate['native_smoke_plan']
    plan = tick.load(closure.artifact(ROOT, original_ref))
    program = closure.artifact(ROOT, plan['program'])
    source = program.read_text(encoding='utf-8')
    frozen = ast.literal_eval(ast.parse(source).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    original_frozen = copy.deepcopy(frozen)
    verified_data = dict(path='pipeline/checks/server1_data_verified_63b27dd8efe04a0bab6dc0291cefdbf0.json',sha256='8c590fa828fbc6c0df840c21ba88b1802cc0d2323eb0a124a57aa328b0c2361f')
    closure.artifact(ROOT,verified_data)
    live = observe(server)
    # This is preparation; a busy card remains busy and is never activated here.
    idle_now = idle(live)
    frozen.update(runtime_server='server-1', runtime_power_limit_watts=350.0,
        runtime_path_mappings=copy.deepcopy(server['runtime_path_mappings']),
        resource_authorization=ref('pipeline/authorizations/server1_gpu0_20261025.json'))
    assert frozen['variant'] == original_frozen['variant']
    assert frozen['goal_sha256'] == original_frozen['goal_sha256']
    body = adapted_body('\n'.join(source.splitlines()[1:]) + '\n', 350.0)
    base = ROOT / ('pipeline/checks/y69_server1_alternative_' + token)
    target = base.with_name(base.name + '_program.py')
    plan_path = base.with_name(base.name + '_plan.json')
    binding = base.with_name(base.name + '_binding.json')
    test_log = base.with_name(base.name + '_tests.log')
    assert not any(p.exists() for p in (target, plan_path, binding, test_log))
    tested = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_y69_server1_alternative.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    test_log.write_bytes(tested.stdout + tested.stderr)
    assert tested.returncode == 0 and b'Ran 8 tests' in test_log.read_bytes()
    assert body.replace(ADAPT, '').replace("== 'dl02-MS-7B73'", "== 'dl01-MS-7C60'") == source.partition('\n')[2]
    target.write_text('FROZEN=' + repr(frozen) + '\n' + body, encoding='utf-8', newline='\n')
    new_plan = copy.deepcopy(plan)
    new_plan.update(program=ref(target), server='server-1', gpu=0, previous_plan=original_ref,
        runtime_power_limit_watts=350.0, resource_authorization=frozen['resource_authorization'],
        required_resource='Fresh exclusive idle authorized server-1 GPU0, no queue/running reservation '
            'or compute PID, exact hostname and approved private runtime/data/path mappings; keep default350W.',
        planned_command='python -B scripts/run_y69_server1_native_probe.py AdaRES-Y69 --token <CURRENT_TOKEN>',
        activation_required=True, native_executed=False)
    tick.save(plan_path, new_plan)
    assert validate_frozen(candidate, new_plan, frozen) == 7
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    current = next(t for t in monitor.load_queue(str(ROOT))['gpu_queues']['server-1'] if t['id'] == 'AdaRES-Y66')
    assert current['status'] == 'running' and current['launch_verified']
    closure.admission(ROOT, current)
    tick.save(binding, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
        previous_plan=original_ref, inactive_server1_plan=ref(plan_path), actual_server1_observation=live,
        actual_idle=idle_now, preparation_only=True, activated=False,
        original_variant_and_YAML_identical=True, goal_and_code_commit_unchanged=True,
        original_diagnostic_body_reversible_exact=True,exact_original_data_verification=verified_data,
        original_guard_tests_preserved=True, alternate_guard_tests_passed=8, test_log=ref(test_log),
        source_and_path_changes='Only explicit host/power guards and deterministic configured path adaptation.',
        native_executed=False, hardware_changed=False, production_sync=False,
        queue_changed=False, running_changed=False, Y66_admission_preserved=True))
    candidate['native_smoke_plan_server1'] = ref(plan_path)
    candidate['evidence'].extend([ref(target), ref(plan_path), ref(binding), ref(test_log), ref(__file__),
        ref('scripts/run_y69_server1_native_probe.py'), ref('scripts/test_y69_server1_alternative.py'), verified_data])
    candidate['next_action'] = ('Use original frozen plan on genuinely free server-2 GPU0; if server-1 GPU0 '
        'becomes free first, verify Y66 canonical terminal archive and run guarded activate on the stored '
        'server1 alternative, then its native two-step runner. No current activation or GPU test. '
        'Only actual native PASS plus original frozen full post-training protocol permits admission/dispatch.')
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=candidate['id'], kind='implementation_test', result='passed',
        command='python -B scripts/prepare_y69_server1_alternative.py prepare --token ' + token,
        observation='Inactive resource-only alternative frozen with identical model/YAML, current grant and '
            '8 path/host/power/idle guards passed. Y66 remains live; no native GPU test, activation or admission.',
        evidence=[ref(binding), ref(test_log)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(binding=ref(binding), inactive_plan=ref(plan_path), tests=8,
                         actual_idle=idle_now, activated=False, native_executed=False)))


def activate(token, server, work, candidate):
    plan_ref = candidate['native_smoke_plan_server1']
    plan = tick.load(closure.artifact(ROOT, plan_ref))
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    assert frozen['runtime_server'] == 'server-1' and frozen['runtime_power_limit_watts'] == 350.0
    assert frozen['runtime_path_mappings'] == server['runtime_path_mappings']
    closure.artifact(ROOT, frozen['resource_authorization'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip() == candidate['code_commit']
    queue = monitor.load_queue(str(ROOT))
    assert not any(t['status'] in ('running', 'claiming') and str(t.get('gpu')) == '0'
                   for t in queue['gpu_queues']['server-1']), 'Canonical terminal archive must precede activation'
    assert not monitor.load_running(str(ROOT)).get('server-1', {}).get('0')
    observation = observe(server)
    assert idle(observation), 'GPU0 remains busy; preserve existing/foreign tasks'
    out = ROOT / ('pipeline/checks/y69_server1_activation_' + token + '.json')
    assert not out.exists()
    tick.save(out, dict(complete=True, token=token, id=candidate['id'], observed_at=datetime.now().astimezone().isoformat(),
        observation=observation, previous_plan=candidate['native_smoke_plan'], active_plan=plan_ref,
        no_queue_or_running_reservations=True, exclusive_idle_observed=True, native_executed=False,
        GPU_lease_and_idle_must_be_rechecked_by_runner=True, hardware_changed=False))
    candidate.update(stage='validation', required_server='server-1', required_gpu=0, native_smoke_plan=plan_ref)
    for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition', 'occupation_evidence'):
        candidate.pop(key, None)
    candidate['evidence'].append(ref(out))
    candidate['next_action'] = plan['planned_command'] + '; only actual two-step native PASS and frozen full evaluation protocol allow admission.'
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(activation=ref(out), active_plan=plan_ref, native_executed=False)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=('prepare', 'activate'))
    p.add_argument('--token', required=True)
    args = p.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-1']
        closure.validate_adares_server(ROOT, server)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y69')
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        assert not candidate.get('admission_receipt') and candidate['required_server'] == 'server-2'
        (prepare if args.operation == 'prepare' else activate)(args.token, server, work, candidate)
    finally:
        lock.close()


if __name__ == '__main__': main()
