"""Retain only the seven tested independent Y91 files and GPU0-only native plans."""
import argparse
import ast
import base64
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from prepare_y91_hidden_confidence_gate import ROOT, CID, files, lease, monitor, ref, tick
from y91_contract import validate_cpu
from y88_deployment import Y88, contract
from check_ef_execution import validate_frozen
from prepare_y69_server1_alternative import adapted_body
from y82_server1_alternative import DATA_PROOF
import closure

HEAD = Y88
BRANCH = 'refs/heads/adares-y91-hidden-confidence-gate'


def git(*args, env=None, data=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, env=env, input=data, timeout=30)


def retain(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == CID)
    assert candidate['stage'] == 'implementation' and not candidate.get('code_commit')
    result = ROOT / ('pipeline/analyses/y91_cpu_' + token + '.json')
    cpu = tick.load(result)
    validate_cpu(cpu)
    program = ROOT / ('pipeline/checks/y91_cpu_' + token + '_program.py')
    payload = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert payload['goal_sha256'] == cpu['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    for path in files():
        value = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert ref(path)['sha256'] == value['sha256'] and path.read_bytes() == base64.b64decode(value['base64'])
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    paths = [path.relative_to(ROOT).as_posix() for path in files()]
    assert not git('ls-files', '--', *paths).strip()
    assert subprocess.run(['git', 'show-ref', '--verify', '--quiet', BRANCH], cwd=ROOT).returncode == 1
    qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
    queue = monitor.load_queue(str(ROOT))
    for name, cid, pid in (('server-1', 'AdaRES-Y88', 3527278), ('server-2', 'AdaRES-Y87', 26281)):
        task = next(task for task in queue['gpu_queues'][name] if task['id'] == cid)
        assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid
        closure.admission(ROOT, task)
        contract(task['code_commit'], HEAD, cid)
    earlier = {cid: copy.deepcopy(next(item for item in work['candidates'] if item['id'] == cid))
               for cid in ('AdaRES-Y89', 'AdaRES-Y90')}
    for cid, prior in earlier.items():
        assert prior['stage'] == 'blocked' and prior['blocker_kind'] == 'exclusive_gpu_slot'
        closure.artifact(ROOT, prior['source_manifest'])
        assert git('rev-parse', prior['retained_branch']).decode().strip() == prior['code_commit']
    base = ROOT / ('pipeline/checks/y91_cpu_version_' + token)
    source = base.with_name(base.name + '_source_snapshot.json')
    index, output = base.with_suffix('.index'), base.with_suffix('.json')
    assert not any(path.exists() for path in (source, index, output))
    actual_index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not actual_index.is_absolute():
        actual_index = ROOT / actual_index
    original_index = hashlib.sha256(actual_index.read_bytes()).hexdigest()
    tick.save(source, dict(complete=True, id=CID, token=token, CPU_tested_payload=payload,
        CPU_result=ref(result), CPU_log=ref(result.with_suffix('.log')), CPU_program=ref(program),
        exact_files=[ref(path) for path in files()], working_HEAD=HEAD))
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    git('read-tree', HEAD, env=env)
    for path in files():
        blob = git('hash-object', '-w', '--stdin', data=path.read_bytes()).decode().strip()
        git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + path.relative_to(ROOT).as_posix(), env=env)
    tree = git('write-tree', env=env).decode().strip()
    delta = [row.split('\t') for row in git('diff-tree', '-r', '--no-commit-id', '--name-status', HEAD, tree).decode().splitlines()]
    assert sorted(delta) == sorted([['A', path] for path in paths])
    commit = git('commit-tree', tree, '-p', HEAD, '-m',
                 'Add Y91 internal first-to-second hidden confidence gate').decode().strip()
    git('update-ref', BRANCH, commit, '0' * 40)
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    assert hashlib.sha256(actual_index.read_bytes()).hexdigest() == original_index
    assert all(path.read_bytes() == git('show', commit + ':' + path.relative_to(ROOT).as_posix()) for path in files())
    assert earlier == {cid: next(item for item in work['candidates'] if item['id'] == cid) for cid in earlier}
    tick.save(output, dict(complete=True, id=CID, code_commit=commit, parent_commit=HEAD,
        retained_branch=BRANCH, working_HEAD=HEAD, main_index_unchanged=True, source_snapshot=ref(source),
        CPU_result=ref(result), CPU_program=ref(program), exact_delta=delta, isolated_index=ref(index),
        earlier_Y89_Y90_preserved=True, no_untested_parent_candidate_included=True,
        native_executed=False, admitted=False, production_sync=False, helper=ref(__file__)))
    candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(output),
        implementation_test_passed=True, CPU_result=ref(result), CPU_program=ref(program))
    candidate['evidence'].extend([
        ref(output), ref(source), ref(index), ref(result), ref(result.with_suffix('.log')), ref(__file__)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
    print(json.dumps(dict(version=ref(output), code_commit=commit, HEAD_preserved=HEAD)))


def plan(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == CID)
    assert candidate['stage'] == 'implementation' and candidate['implementation_test_passed']
    assert not candidate.get('native_preparation') and git('rev-parse', 'HEAD').decode().strip() == HEAD
    version = tick.load(closure.artifact(ROOT, candidate['source_manifest']))
    bundle = tick.load(closure.artifact(ROOT, version['source_snapshot']))
    payload = bundle['CPU_tested_payload']
    validate_cpu(tick.load(closure.artifact(ROOT, version['CPU_result'])))
    assert git('rev-parse', BRANCH).decode().strip() == candidate['code_commit']
    base = ROOT / ('pipeline/checks/y91_native_' + token)
    tests, output = base.with_name(base.name + '_guards.log'), base.with_suffix('.json')
    assert not tests.exists() and not output.exists()
    response = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_y91_preparation.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    tests.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and b'Ran 10 tests' in tests.read_bytes()
    authority = ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')
    assert authority['sha256'] == '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
    closure.artifact(ROOT, DATA_PROOF)
    servers = monitor.load_servers(str(ROOT))
    assert servers['server-6']['allowed_gpus'] == []
    body = closure.artifact(ROOT, candidate['prepared_native_body']).read_text(encoding='utf-8')
    qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
    plans, programs = {}, []
    for name, power in (('server-2', 280.), ('server-1', 350.)):
        server = servers[name]
        closure.validate_adares_server(ROOT, server)
        assert server['allowed_gpus'] == [0]
        grant = authority if name == 'server-2' else ref('pipeline/authorizations/server1_gpu0_20261025.json')
        frozen = dict(variant=payload['variant'], code_commit=candidate['code_commit'],
            goal_sha256=payload['goal_sha256'], train_source_lf_sha256=payload['train_source_lf_sha256'],
            runtime_server=name, runtime_physical_gpu=0, runtime_power_limit_watts=power, resource_authorization=grant)
        program_body = body
        if name == 'server-1':
            frozen['runtime_path_mappings'] = copy.deepcopy(server['runtime_path_mappings'])
            program_body = adapted_body(body, power)
        suffix = name.replace('-', '')
        program = base.with_name(base.name + '_' + suffix + '_program.py')
        ppath = base.with_name(base.name + '_' + suffix + '_plan.json')
        assert not program.exists() and not ppath.exists()
        plan_ = dict(complete=True, id=CID, config=candidate['config'], profile='experiment_first',
            code_commit=candidate['code_commit'], goal_sha256=frozen['goal_sha256'], server=name, gpu=0,
            runtime_power_limit_watts=power, resource_authorization=grant, working_HEAD_preserved=HEAD,
            retained_branch=BRANCH, native_executed=False, admitted=False, native_PASS_required=True,
            guard_tests_passed=10, Python38_compile=True,
            planned_command='python -B scripts/run_y91_native.py --server ' + name + ' --token <CURRENT_TOKEN>',
            required_resource='Fresh exclusive idle/unreserved matching authorized GPU0 after original owner exact fixed30 canonical archive; no GPU1 or power/data/runtime substitution.',
            checks='Original native TRAIN two finite task losses/backward/updates; fourteen exact original query/mask/mass/evidence, first/second hidden gate and final linear consumers; all70 original MLP layer calls, original gates, zero then learned hidden response, finite original/new/input gradients, modes/memory.')
        assert validate_frozen(candidate, plan_, frozen) == 7
        for field in ('goal', 'source'):
            bad = copy.deepcopy(frozen)
            if field == 'goal':
                bad['goal_sha256'] = '0' * 64
            else:
                bad['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
            try:
                validate_frozen(candidate, plan_, bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('Corrupt frozen input accepted')
        program.write_text('FROZEN=' + repr(frozen) + '\n' + program_body, encoding='utf-8', newline='\n')
        ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
        plan_['program'] = ref(program)
        tick.save(ppath, plan_)
        plans[name] = ref(ppath)
        programs.append(ref(program))
    tick.save(output, dict(complete=True, id=CID, token=token, code_commit=candidate['code_commit'],
        CPU_version=candidate['source_manifest'], CPU_result=version['CPU_result'], CPU_program=version['CPU_program'],
        native_plans=plans, guard_tests=ref(tests), guard_tests_passed=10,
        exact_callback_functions_and_classes=15, CPU_native_callbacks_actually_executed=True,
        observer_core=candidate['observer_core'], stale_goal_refused=True, tampered_source_refused=True,
        original_data_proof=DATA_PROOF, earlier_Y89_Y90_preserved=True, native_executed=False,
        admitted=False, production_sync=False, helper=ref(__file__)))
    candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        required_server='server-2', required_gpu=0, resource_work_kind='native_smoke',
        native_smoke_plan=plans['server-2'], native_smoke_plan_server2_gpu0=plans['server-2'],
        native_smoke_plan_server1=plans['server-1'], native_preparation=ref(output),
        CPU_native_observer_verification=ref(output),
        blocker='Original independent Y87/Y88 reserve both authorized GPU0 resources. Preserve them and earlier Y89/Y90; native task needs an exclusive matching unreserved GPU0.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Original matching owner naturally complete, exact fixed30 checkpoint-verified/canonically archived; matching GPU0 physically idle with no compute PID or reservation. Never GPU1.',
        next_action='Do not replay CPU/version/plan checks. Earlier Y89/Y90 precede Y91; preserve production HEADY88 and original runs. Only a fresh matching authorized exclusive idle GPU0 permits the stored original native YAML/TRAIN two-step program via run_y91_native.py. Actual native PASS required for full original frozen review/receipt/admission and safe additive source selection/sync/dispatch with increasing counters. If HEAD advances, re-retain identical tested seven files additively with explicit identity binding; never reset or combine mechanisms. GPU1/Y57 remain paused.')
    candidate['evidence'].extend([ref(output), ref(tests), authority, DATA_PROOF,
        ref('scripts/y91_contract.py'), ref('scripts/test_y91_preparation.py')] + list(plans.values()) + programs)
    work['tick_attempt'] = dict(
        tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y91_cpu ' + version['CPU_program']['path'] + ' --token ' + token + ' --seconds 480',
        observation='Actual original CPU YAML/full model/backbone/TRAIN loading, scalar/bounded/no-RNG/permutation controls, seven-stage W2/input-gradient anchors, confidence-only intervention/bypass and exact original hidden/evidence consumers/two synthetic updates passed on first attempt. Ten actual-evidence guards passed; seven files retained independently and matching GPU0-only plans frozen. Both GPUs busy; no native GPU PASS/admission/efficacy.',
        evidence=[version['CPU_result'], candidate['source_manifest'], ref(output)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
    print(json.dumps(dict(preparation=ref(output), plans=plans, tests=10, native_executed=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('retain', 'plan'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        dict(retain=retain, plan=plan)[args.operation](args.token)
    finally:
        writer.close()
