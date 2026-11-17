"""Retain tested additive Y88 files and a server1-only native plan."""
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
from prepare_y88_mask_query_quadratic import ROOT, CID, files, lease, monitor, ref, tick
from y88_contract import validate_cpu
from y87_deployment import Y87, contract
from check_ef_execution import validate_frozen
from prepare_y69_server1_alternative import adapted_body
import closure

HEAD = Y87
BRANCH = 'refs/heads/adares-y88-mask-query-quadratic'


def git(*args, env=None, data=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, env=env, input=data, timeout=30)


def retain(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == CID)
    assert candidate['stage'] == 'implementation' and not candidate.get('code_commit')
    result = ROOT / ('pipeline/analyses/y88_cpu_' + token + '.json')
    cpu = tick.load(result)
    validate_cpu(cpu)
    program = ROOT / ('pipeline/checks/y88_cpu_' + token + '_program.py')
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
    for name, cid, pid in (('server-1', 'AdaRES-Y84', 243952), ('server-2', 'AdaRES-Y87', 26281)):
        task = next(task for task in queue['gpu_queues'][name] if task['id'] == cid)
        assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid
        closure.admission(ROOT, task)
        contract(task['code_commit'], HEAD, cid)
    base = ROOT / ('pipeline/checks/y88_cpu_version_' + token)
    source, index, output = base.with_name(base.name + '_source_snapshot.json'), base.with_suffix('.index'), base.with_suffix('.json')
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
    commit = git('commit-tree', tree, '-p', HEAD, '-m', 'Add Y88 zero-anchored quadratic mask-query readout').decode().strip()
    git('update-ref', BRANCH, commit, '0' * 40)
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    assert hashlib.sha256(actual_index.read_bytes()).hexdigest() == original_index
    assert all(path.read_bytes() == git('show', commit + ':' + path.relative_to(ROOT).as_posix()) for path in files())
    tick.save(output, dict(complete=True, id=CID, code_commit=commit, parent_commit=HEAD,
        retained_branch=BRANCH, working_HEAD=HEAD, main_index_unchanged=True, source_snapshot=ref(source),
        CPU_result=ref(result), CPU_program=ref(program), exact_delta=delta, isolated_index=ref(index),
        native_executed=False, admitted=False, production_sync=False, helper=ref(__file__)))
    candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(output),
        implementation_test_passed=True, CPU_result=ref(result), CPU_program=ref(program))
    candidate['evidence'].extend([ref(output), ref(source), ref(index), ref(result), ref(result.with_suffix('.log')), ref(__file__)])
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
    base = ROOT / ('pipeline/checks/y88_native_' + token)
    program, ppath = base.with_name(base.name + '_server1_program.py'), base.with_name(base.name + '_server1_plan.json')
    tests, output = base.with_name(base.name + '_guards.log'), base.with_suffix('.json')
    assert not any(path.exists() for path in (program, ppath, tests, output))
    response = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_y88_native_plan.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    tests.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and b'Ran 8 tests' in tests.read_bytes()
    server = monitor.load_servers(str(ROOT))['server-1']
    closure.validate_adares_server(ROOT, server)
    assert server['allowed_gpus'] == [0]
    authority = ref('pipeline/authorizations/server1_gpu0_20261025.json')
    closure.artifact(ROOT, authority)
    frozen = dict(variant=payload['variant'], code_commit=candidate['code_commit'], goal_sha256=payload['goal_sha256'],
        train_source_lf_sha256=payload['train_source_lf_sha256'], runtime_server='server-1',
        runtime_physical_gpu=0, runtime_power_limit_watts=350., resource_authorization=authority,
        runtime_path_mappings=copy.deepcopy(server['runtime_path_mappings']))
    plan_ = dict(complete=True, id=CID, config=candidate['config'], profile='experiment_first',
        code_commit=candidate['code_commit'], goal_sha256=frozen['goal_sha256'], server='server-1', gpu=0,
        runtime_power_limit_watts=350., resource_authorization=authority, runtime_path_mappings=frozen['runtime_path_mappings'],
        working_HEAD_preserved=HEAD, retained_branch=BRANCH, native_executed=False, admitted=False,
        native_PASS_required=True, guard_tests_passed=8, Python38_compile=True,
        planned_command='python -B scripts/run_y88_server1_native.py --token <CURRENT_TOKEN>',
        required_resource='Fresh exclusive idle/unreserved server1GPU0/350W after exact Y84 fixed30 canonical archive. No GPU1.',
        checks='Original native TRAIN two finite losses/backward/updates, fourteen exact quadratic projections and mask consumers, original score/indicator/query_norm/coarse gates, original/new/input finite gradients, native modes/memory. No efficacy from smoke.')
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
    original_body = closure.artifact(ROOT, candidate['prepared_native_body']).read_text(encoding='utf-8')
    body = adapted_body(original_body, 350.)
    program.write_text('FROZEN=' + repr(frozen) + '\n' + body, encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    plan_['program'] = ref(program)
    tick.save(ppath, plan_)
    tick.save(output, dict(complete=True, id=CID, token=token, code_commit=candidate['code_commit'],
        CPU_version=candidate['source_manifest'], CPU_result=version['CPU_result'], CPU_program=version['CPU_program'],
        native_plans={'server-1:0': ref(ppath)}, guard_tests=ref(tests), guard_tests_passed=8,
        exact_callback_functions_and_classes=11, CPU_native_callbacks_actually_executed=True,
        observer_core=ref('scripts/y88_observer_core.py'), stale_goal_refused=True, tampered_source_refused=True,
        native_executed=False, admitted=False, production_sync=False, helper=ref(__file__)))
    candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        required_server='server-1', required_gpu=0, resource_work_kind='native_smoke',
        native_smoke_plan=ref(ppath), native_smoke_plan_server1=ref(ppath),
        native_preparation=ref(output), CPU_native_observer_verification=ref(output),
        blocker='Original Y84 still reserves server1GPU0; preserve its exact running task until natural terminal verification and archival.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Exact Y84 naturally completed and canonically archived; fresh exclusive server1GPU0/350W idle and unreserved with no compute PID.',
        next_action='Do not replay CPU/version/plan checks. On fresh exclusive server1GPU0/350W run the exact original native TRAIN two-step program; actual PASS required before full frozen review protocol, receipt/admission, safe source selection/sync, canonical dispatch and increasing counters. No GPU1, Y57 continuation or efficacy claim.')
    candidate['evidence'].extend([ref(output), ref(ppath), ref(program), ref(tests), authority,
                                 ref('scripts/y88_contract.py'), ref('scripts/test_y88_native_plan.py')])
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y88_cpu ' + version['CPU_program']['path'] + ' --token ' + token + ' --seconds 480',
        observation='Actual CPU native YAML/model/backbone/TRAIN collation, independent scalar/zero/RNG/permutation controls, seven-stage W2/input-gradient anchors and exact native observers/two synthetic updates passed. Exact seven files retained; server1GPU0-only native plan prepared with eight rejection guards. Not native GPU PASS or efficacy.',
        evidence=[version['CPU_result'], candidate['source_manifest'], ref(output)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(preparation=ref(output), plan=ref(ppath), tests=8, native_executed=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('retain', 'plan'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 240)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        dict(retain=retain, plan=plan)[args.operation](args.token)
    finally:
        lock.close()
