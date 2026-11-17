"""Retain seven tested Y87 files and a GPU0-only plan; never admit from CPU checks."""
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
from prepare_y87_self_language_product import ROOT, CID, files, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from y85_deployment import Y85, contract
import closure

BRANCH = 'refs/heads/adares-y87-self-language-product'
PARENT = '2d2a0b98865977a6e2f37d316979344ffc56817c'


def git(*args, env=None, data=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, env=env, input=data, timeout=30)


def retain(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == CID)
    assert candidate['stage'] == 'implementation' and not candidate.get('code_commit')
    result = ROOT / ('pipeline/analyses/y87_cpu_' + token + '.json')
    cpu = tick.load(result)
    flags = ('complete', 'implementation_test_passed', 'full_model_CPU_constructed',
        'configured_TRAIN_data_collated', 'released_backbone_loaded', 'native_YAML_semantically_unchanged',
        'train_eval_train_roundtrip', 'decoder_initial_RNG_and_parent_weights_exact',
        'decoder_all_stages_anchor_exact', 'decoder_input_gradients_anchor_close',
        'nonzero_fusion_changes_refined_and_final_masks', 'bypassed_fusion_negative_control_exact', 'original_shared_head_query_norm_consumption_verified',
        'static_mask_and_original_hard_gate_preserved', 'original_self_language_and_norm_ffn_consumption_verified')
    assert all(cpu[key] is True for key in flags)
    assert cpu['id'] == CID and (cpu['parameters'], cpu['additional_parameters']) == (164695173, 65536)
    assert cpu['synthetic_CPU_decoder_optimizer_steps'] == 2 and len(cpu['CPU_fusion_paths']) == 12
    assert all(cpu[key] is False for key in
               ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward', 'native_sparse_forward', 'validation_used', 'admission'))
    assert {row['dtype'] for row in cpu['component_controls']} == {'torch.float32', 'torch.float64'}
    assert all(value is True for row in cpu['component_controls'] for key, value in row.items() if key != 'dtype')
    for index, row in enumerate(cpu['CPU_decoder_steps']):
        assert row['step'] == index and row['all_finite'] and row['indicator_tensors'] == 10
        assert row['indicator_calls'] == row['query_norm_consumption_calls'] == 7
        assert row['fusion_calls'] == row['original_norm_consumption_calls'] == row['original_ffn_consumption_calls'] == 6
        assert row['both_input_gradients_finite'] is True
        assert row['new_gradient_max'] > 0 and row['parameter_update_max'] > 0
    program = ROOT / ('pipeline/checks/y87_cpu_' + token + '_program.py')
    payload = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert payload['goal_sha256'] == cpu['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    for path in files():
        value = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert ref(path)['sha256'] == value['sha256']
        assert path.read_bytes() == base64.b64decode(value['base64'])
    assert git('rev-parse', 'HEAD').decode().strip() == Y85
    paths = [path.relative_to(ROOT).as_posix() for path in files()]
    assert not git('ls-files', '--', *paths).strip()
    assert subprocess.run(['git', 'show-ref', '--verify', '--quiet', BRANCH], cwd=ROOT).returncode == 1
    qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
    queue = monitor.load_queue(str(ROOT))
    for name, cid, pid in [('server-1', 'AdaRES-Y84', 243952), ('server-2', 'AdaRES-Y85', 22090)]:
        task = next(task for task in queue['gpu_queues'][name] if task['id'] == cid)
        assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid
        closure.admission(ROOT, task)
        contract(task['code_commit'], Y85, cid)
    base = ROOT / ('pipeline/checks/y87_cpu_version_' + token)
    source, index, output = base.with_name(base.name + '_source_snapshot.json'), base.with_suffix('.index'), base.with_suffix('.json')
    assert not any(path.exists() for path in (source, index, output))
    actual_index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not actual_index.is_absolute():
        actual_index = ROOT / actual_index
    before = hashlib.sha256(actual_index.read_bytes()).hexdigest()
    tick.save(source, dict(complete=True, id=CID, token=token, CPU_tested_payload=payload,
        CPU_result=ref(result), CPU_log=ref(result.with_suffix('.log')), CPU_program=ref(program),
        exact_files=[ref(path) for path in files()], working_HEAD=Y85))
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    assert git('rev-parse', PARENT + '^').decode().strip() == Y85
    previous = next(item for item in work['candidates'] if item['id'] == 'AdaRES-Y86')
    assert previous['code_commit'] == PARENT and previous['implementation_test_passed']
    closure.artifact(ROOT, previous['source_manifest'])
    prior_delta = [row.split('\t') for row in git('diff', '--no-renames', '--name-status', Y85, PARENT).decode().splitlines()]
    assert len(prior_delta) == 7 and all(status == 'A' and 'adares_y86_overlap_readout_skip' in path for status, path in prior_delta)
    assert all((ROOT / path).read_bytes() == git('show', PARENT + ':' + path) for _, path in prior_delta)
    git('read-tree', PARENT, env=env)
    for path in files():
        blob = git('hash-object', '-w', '--stdin', data=path.read_bytes()).decode().strip()
        git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + path.relative_to(ROOT).as_posix(), env=env)
    tree = git('write-tree', env=env).decode().strip()
    delta = [row.split('\t') for row in git('diff-tree', '-r', '--no-commit-id', '--name-status', PARENT, tree).decode().splitlines()]
    assert sorted(delta) == sorted([['A', path] for path in paths])
    commit = git('commit-tree', tree, '-p', PARENT, '-m', 'Add Y87 self-language query product before original norm').decode().strip()
    git('update-ref', BRANCH, commit, '0' * 40)
    assert git('rev-parse', 'HEAD').decode().strip() == Y85
    assert hashlib.sha256(actual_index.read_bytes()).hexdigest() == before
    assert all(path.read_bytes() == git('show', commit + ':' + path.relative_to(ROOT).as_posix()) for path in files())
    tick.save(output, dict(complete=True, id=CID, code_commit=commit, parent_commit=PARENT, retained_branch=BRANCH,
        working_HEAD=Y85, main_index_unchanged=True, source_snapshot=ref(source), CPU_result=ref(result),
        CPU_program=ref(program), exact_delta=delta, isolated_index=ref(index),
        native_executed=False, admitted=False, production_sync=False, helper=ref(__file__)))
    candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(output),
                     implementation_test_passed=True)
    candidate['evidence'].extend([ref(output), ref(source), ref(index), ref(result), ref(result.with_suffix('.log')), ref(__file__)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
    print(json.dumps(dict(version=ref(output), code_commit=commit, HEAD_preserved=Y85, native_executed=False)))


def plan(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == CID)
    assert candidate['stage'] == 'implementation' and candidate['implementation_test_passed']
    assert not candidate.get('native_preparation')
    version = tick.load(closure.artifact(ROOT, candidate['source_manifest']))
    bundle = tick.load(closure.artifact(ROOT, version['source_snapshot']))
    payload = bundle['CPU_tested_payload']
    assert git('rev-parse', 'HEAD').decode().strip() == Y85
    assert git('rev-parse', BRANCH).decode().strip() == candidate['code_commit']
    core = ROOT / 'scripts/y87_observer_core.py'
    body = closure.artifact(ROOT, candidate['prepared_native_body']).read_text(encoding='utf-8')
    expected = {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(core.read_text(encoding='utf-8')).body if isinstance(node, ast.FunctionDef)}
    assert len(expected) == 14
    for source in (body, closure.artifact(ROOT, version['CPU_program']).read_text(encoding='utf-8')):
        assert {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name in expected} == expected
    base = ROOT / ('pipeline/checks/y87_native_' + token)
    tests, program = base.with_name(base.name + '_guards.log'), base.with_name(base.name + '_gpu0_program.py')
    ppath, output = base.with_name(base.name + '_gpu0_plan.json'), base.with_suffix('.json')
    assert not any(path.exists() for path in (tests, program, ppath, output))
    response = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
        '-s', 'scripts', '-p', 'test_y87_native_plan.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    tests.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and b'Ran 9 tests' in tests.read_bytes()
    authority = ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')
    assert authority['sha256'] == '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
    assert monitor.load_servers(str(ROOT))['server-2']['allowed_gpus'] == [0]
    frozen = dict(variant=payload['variant'], code_commit=candidate['code_commit'],
        goal_sha256=payload['goal_sha256'], train_source_lf_sha256=payload['train_source_lf_sha256'],
        runtime_server='server-2', runtime_physical_gpu=0, runtime_power_limit_watts=280.,
        resource_authorization=authority)
    plan_ = dict(complete=True, id=CID, config=candidate['config'], profile='experiment_first',
        code_commit=candidate['code_commit'], goal_sha256=frozen['goal_sha256'], server='server-2',
        gpu=0, runtime_power_limit_watts=280., resource_authorization=authority,
        working_HEAD_preserved=Y85, retained_branch=BRANCH, native_executed=False, admitted=False,
        native_PASS_required=True, guard_tests_passed=9, Python38_compile=True,
        planned_command='python -B scripts/run_experiment_first_probe.py y87_gpu0_smoke '
            + program.relative_to(ROOT).as_posix() + ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id 0',
        required_resource='Fresh exclusive idle/unreserved server-2 GPU0/280W after exact Y85 terminal verification/archive; preserve Y84 and both user pauses.',
        checks='Original YAML/TRAIN two finite losses/backward/optimizer, twelve exact original self/language sum-product paths and original norm/FFN consumers, zero then nonzero response, original/new/input gradients and updates, original masks/gates/head/norms/native modes and memory.')
    assert validate_frozen(candidate, plan_, frozen) == 7
    for field in ('goal', 'source'):
        bad = copy.deepcopy(frozen)
        if field == 'goal':
            bad['goal_sha256'] = '0' * 64
        else:
            bad['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
        refused = False
        try:
            validate_frozen(candidate, plan_, bad)
        except AssertionError:
            refused = True
        assert refused
    program.write_text('FROZEN=' + repr(frozen) + '\n' + body, encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    plan_['program'] = ref(program)
    tick.save(ppath, plan_)
    tick.save(output, dict(complete=True, id=CID, token=token, code_commit=candidate['code_commit'],
        CPU_version=candidate['source_manifest'], CPU_result=version['CPU_result'], CPU_program=version['CPU_program'],
        native_plans={'0': ref(ppath)}, guard_tests=ref(tests), guard_tests_passed=9,
        exact_callback_functions_and_classes=14, CPU_native_callbacks_actually_executed=True,
        observer_core=ref(core), stale_goal_refused=True, tampered_source_refused=True,
        native_executed=False, admitted=False, production_sync=False, helper=ref(__file__)))
    candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        required_server='server-2', required_gpu=0, resource_work_kind='native_smoke',
        native_smoke_plan=ref(ppath), native_smoke_plan_server2_gpu0=ref(ppath),
        native_preparation=ref(output), CPU_native_observer_verification=ref(output), resource_pause=authority,
        blocker='Authorized GPU0 resources are reserved by original independent Y84/Y85. Preserve both; actual native task requires an exclusive authorized idle unreserved GPU0.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Exact original owner naturally completed and canonically archived; fresh exclusive server-2 GPU0/280W idle with no compute or reservation.',
        next_action='Do not replay CPU/version/native-plan preparation. Y86 precedes Y87. Only fresh exclusive authorized GPU0 permits this exact native YAML/TRAIN two-step task. Actual PASS is required before full original frozen review plan, receipt/admission/safe source selection/sync/dispatch and increasing native counters. GPU1 forbidden; no efficacy from CPU checks.')
    candidate['evidence'].extend([ref(output), ref(ppath), ref(program), ref(tests), authority, ref(core),
        candidate['prepared_native_body'], ref('scripts/test_y87_native_plan.py'), ref(__file__)])
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y87_cpu pipeline/checks/y87_cpu_' + token + '_program.py --token ' + token + ' --seconds 480',
        observation='Actual CPU native model/backbone/TRAIN loading, independent scalar/zero/RNG/permutation controls, seven-stage W2/input-gradient anchors, original self/language/norm/FFN consumers and two synthetic updates passed; exact retained seven-file version and GPU0-only native plan with fourteen identical actually executed callbacks/nine guards. No native GPU PASS/admission/efficacy.',
        evidence=[version['CPU_result'], candidate['source_manifest'], ref(output)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(preparation=ref(output), plan=ref(ppath), native_executed=False, tests=9)))


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
