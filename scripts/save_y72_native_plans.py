"""Save a scoped Y72 branch commit without moving the protected Y71 working HEAD."""
import argparse
import ast
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from prepare_y72_visual_mask_refinement import ROOT, CID, files, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from y68_terminal_contract import build_contract
import closure

HEAD = '87019e7c2f640f99cb82a3e5b396321fc5b54d25'
BRANCH = 'refs/heads/adares-y72-visual-mask-refinement'


def git(*args, env=None, data=None):
    return subprocess.check_output(['git', '-c', 'core.autocrlf=false', *args],
                                   cwd=ROOT, env=env, input=data, timeout=90)


def save(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    assert candidate['stage'] == 'implementation' and candidate.get('code_commit') is None
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    old_snapshot = closure.artifact(ROOT, candidate['source_manifest'])
    checked = tick.load(old_snapshot)
    assert checked['complete'] and checked['id'] == CID and checked['base_head'] == HEAD
    payload = checked['CPU_tested_payload']
    cpu_path = closure.artifact(ROOT, checked['cpu_test'])
    cpu = tick.load(cpu_path)
    assert cpu['implementation_test_passed'] and not cpu['CUDA_initialized'] and not cpu['admission']
    assert cpu['id'] == CID and cpu['config'] == candidate['config']
    assert cpu['decoder_refinement_calls'] == 6 and cpu['decoder_indicator_calls'] == 7
    assert payload['goal_sha256'] == cpu['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    paths = [p.relative_to(ROOT).as_posix() for p in files()]
    for path in files():
        value = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert ref(path)['sha256'] == value['sha256']
    assert not git('ls-files', '--', *paths).strip(), 'Y72 must be new at the protected HEAD'
    assert subprocess.run(['git', 'show-ref', '--verify', '--quiet', BRANCH], cwd=ROOT).returncode == 1
    base = 'pipeline/checks/y72_native_' + token
    outputs = {name: ROOT / (base + suffix) for name, suffix in (
        ('inputs', '_inputs.json'), ('tests', '_guards.log'), ('commit', '_commit.json'),
        ('version', '_version.json'), ('index', '_isolated.index'))}
    destinations = [ROOT / (base + '_gpu' + str(gpu) + suffix)
                    for gpu in (0, 1) for suffix in ('_plan.json', '_program.py')]
    assert not any(p.exists() for p in list(outputs.values()) + destinations)
    body = ROOT / 'scripts/probe_y72_native_smoke.py'
    ast.parse(body.read_text(encoding='utf-8'), feature_version=(3, 8))
    command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
               '-p', 'test_y72_native_smoke_plan.py', '-v']
    tested = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
    outputs['tests'].write_bytes(tested.stdout + tested.stderr)
    assert tested.returncode == 0 and b'Ran 11 tests' in outputs['tests'].read_bytes()
    servers = monitor.load_servers(str(ROOT))
    assert servers['server-2']['allowed_gpus'] == [0, 1]
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    queue = monitor.load_queue(str(ROOT))
    originals = []
    for server, cid, pid in [('server-2', 'AdaRES-Y68', 20742), ('server-2', 'AdaRES-Y70', 52561),
                             ('server-1', 'AdaRES-Y69', 1135633)]:
        task = next(t for t in queue['gpu_queues'][server] if t['id'] == cid)
        assert task['status'] == 'running' and task['launch_verified'] is True and task['pid'] == pid
        closure.admission(ROOT, task)
        if cid == 'AdaRES-Y68':
            build_contract(task, servers[server])
        originals.append(dict(id=cid, pid=pid, code_commit=task['code_commit'], server=server, gpu=task['gpu']))
    paused = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    y71 = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y71')
    y71_plan = tick.load(closure.artifact(ROOT, y71['native_smoke_plan']))
    y71_program = closure.artifact(ROOT, y71_plan['program'])
    y71_frozen = ast.literal_eval(ast.parse(y71_program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(y71, y71_plan, y71_frozen) == 7
    index_path = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    index_before = hashlib.sha256(index_path.read_bytes()).hexdigest()
    authorization = tick.load(ROOT / 'pipeline/closure_policy.json')['experiment_first']['authorization']
    closure.artifact(ROOT, authorization)
    grant = ref('pipeline/authorizations/server2_gpu1_20261029.json')
    assert tick.load(closure.artifact(ROOT, grant))['authorization_granted']
    tick.save(outputs['inputs'], dict(complete=True, id=CID, parent_commit=HEAD,
        CPU_source_snapshot=ref(old_snapshot), CPU_test=ref(cpu_path), exact_files=[ref(p) for p in files()],
        intended_branch=BRANCH, main_index_sha256=index_before, queue_before=queue_before,
        running_before=running_before, prior_tick_attempt=work.get('tick_attempt'),
        authorization=authorization, native_executed=False, admitted=False))
    # Git's alternate index creates exactly seven additions. The working index and HEAD are untouched.
    env = dict(os.environ, GIT_INDEX_FILE=str(outputs['index']))
    git('read-tree', HEAD, env=env)
    for path in files():
        oid = git('hash-object', '-w', '--stdin', data=path.read_bytes()).decode().strip()
        git('update-index', '--add', '--cacheinfo', '100644,' + oid + ',' + path.relative_to(ROOT).as_posix(), env=env)
    tree = git('write-tree', env=env).decode().strip()
    delta = [line.split('\t') for line in git('diff-tree', '-r', '--no-commit-id', '--name-status', HEAD, tree).decode().splitlines()]
    assert sorted(delta) == sorted([['A', path] for path in paths])
    lease(token, 150)
    commit = git('commit-tree', tree, '-p', HEAD, '-m', 'Add Y72 current visual-state mask-feature refinement').decode().strip()
    tick.save(outputs['commit'], dict(complete=True, id=CID, code_commit=commit, parent_commit=HEAD,
        tree=tree, exact_delta=delta, intended_branch=BRANCH, working_HEAD_unchanged=True,
        source_snapshot=ref(old_snapshot), note='Scoped seven-file commit object; no working checkout or production deployment.'))
    git('update-ref', BRANCH, commit, '0' * 40)
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == index_before
    assert git('rev-parse', BRANCH).decode().strip() == commit
    assert git('rev-parse', commit + '^').decode().strip() == HEAD
    checked_candidate = dict(candidate, code_commit=commit)
    plans = {}; programs = []
    for gpu, power in ((0, 280.0), (1, 390.0)):
        frozen = dict(variant=payload['variant'], code_commit=commit, goal_sha256=payload['goal_sha256'],
            train_source_lf_sha256=payload['train_source_lf_sha256'], runtime_physical_gpu=gpu,
            runtime_power_limit_watts=power, resource_authorization=grant)
        program = ROOT / (base + '_gpu' + str(gpu) + '_program.py')
        plan_path = ROOT / (base + '_gpu' + str(gpu) + '_plan.json')
        plan = dict(complete=True, id=CID, config=candidate['config'], profile='experiment_first',
            code_commit=commit, parent_commit=HEAD, retained_branch=BRANCH, goal_sha256=frozen['goal_sha256'],
            server='server-2', gpu=gpu, runtime_power_limit_watts=power, resource_authorization=grant,
            required_resource='Fresh exclusive idle/unreserved authorized server-2 GPU' + str(gpu) +
                '; canonical terminal archive, no compute PID, exact host/private runtime/original data and existing power. '
                'Preserve original Y68/Y69/Y70 and paused Y57. Prefer nearer frozen Y71.',
            planned_command='python -B scripts/run_experiment_first_probe.py y72_gpu' + str(gpu) + '_smoke ' +
                program.relative_to(ROOT).as_posix() + ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id ' + str(gpu),
            checks='Original native YAML/TRAIN/tools.train two finite loss/backward/optimizer steps; six current '
                'ila_ffn identities and original static mask/padding identities, six actual refined prediction consumers, '
                'seven shared indicators, independent scalar formula at first/last expressions and first/middle/last '
                'valid feature rows/channel0/127/255, initial zero/next-step response, padding unchanged, finite new '
                'gradient/actual weight update and source-feature gradients, native modes/peak memory.',
            guard_tests_passed=11, Python38_compile=True, native_executed=False, admitted=False,
            native_PASS_required=True, working_HEAD_preserved=HEAD)
        assert validate_frozen(checked_candidate, plan, frozen) == 7
        for field in ('goal', 'source'):
            bad = copy.deepcopy(frozen)
            if field == 'goal':
                bad['goal_sha256'] = '0' * 64
            else:
                bad['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
            rejected = False
            try:
                validate_frozen(checked_candidate, plan, bad)
            except AssertionError:
                rejected = True
            assert rejected
        program.write_text('FROZEN=' + repr(frozen) + '\n' + body.read_text(encoding='utf-8'),
                           encoding='utf-8', newline='\n')
        ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
        plan['program'] = ref(program)
        tick.save(plan_path, plan)
        plans[str(gpu)] = ref(plan_path); programs.append(ref(program))
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == index_before
    assert validate_frozen(y71, y71_plan, y71_frozen) == 7
    tick.save(outputs['version'], dict(complete=True, id=CID, code_commit=commit, parent_commit=HEAD,
        retained_branch=BRANCH, exact_files=[ref(p) for p in files()], CPU_source_snapshot=ref(old_snapshot),
        CPU_test=ref(cpu_path), guard_tests=ref(outputs['tests']), guard_tests_passed=11,
        commit_record=ref(outputs['commit']), isolated_index=ref(outputs['index']),
        main_index_unchanged=True, working_HEAD_unchanged=True, Y71_seven_files_and_plan_verified=True,
        Y68_terminal_local_source_contract_verified=True, queue_and_running_preserved=True,
        original_admissions_valid=True, original_runs=originals, Y57_paused=True,
        native_plans=plans, native_executed=False, admitted=False, pushed=False,
        production_sync=False, stale_goal_refused=True, tampered_source_refused=True,
        evidence=[ref(__file__), ref(body), ref('scripts/test_y72_native_smoke_plan.py')],
        limitation='CPU guard/version preparation only; native sparse task loss/backward/optimizer and efficacy unobserved.'))
    additions = [ref(p) for p in outputs.values()] + list(plans.values()) + programs + [ref(body), ref(__file__),
                 ref('scripts/test_y72_native_smoke_plan.py')]
    candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        required_server='server-2', required_gpu=0, resource_work_kind='native_smoke', code_commit=commit,
        source_manifest=ref(outputs['version']), retained_branch=BRANCH,
        version_status='CPU-tested scoped seven-file branch commit; working HEAD87019/index unchanged for Y71.',
        native_smoke_plan=plans['0'], native_smoke_plan_server2_gpu0=plans['0'],
        native_smoke_plan_server2_gpu1=plans['1'],
        blocker='Server-2 GPU0 runs original Y68 and GPU1 runs original Y70; server-1 GPU0 runs original Y69. '
                'No exclusive whole GPU is available. Y71 remains nearer to admission; preserve all runs and paused Y57.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Canonical terminal archive and fresh exclusive physically idle/unreserved authorized GPU; select exact matching plan before testing.',
        next_action='Do not replay CPU/model/commit/plan preparation. Preserve Y71 exact working HEAD87019 and prioritize '
                    'its frozen handoff. When an authorized server-2 whole GPU is actually idle/unreserved after canonical '
                    'terminal archive, select its exact Y72 bound plan and verify current goal/source/retained commit. Run '
                    'real native YAML/TRAIN two-step finite loss/backward/optimizer and intended-path checks. Only actual '
                    'PASS permits the original complete frozen evaluation/cost protocol and admission. Before production '
                    'sync, select the retained Y72 commit via safe fast-forward only after preserving/rebinding necessary '
                    'live source/observer/deployment contracts; never loosen old helpers or move/restart protected jobs. '
                    'Then gated same-tick dispatch and increasing native steps. Server-1 requires separately verified '
                    'runtime/path adaptation. Existing scoped authorization applies; Y57 remains independently paused.')
    candidate['evidence'].extend(additions)
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B -m unittest discover -s scripts -p test_y72_native_smoke_plan.py -v '
                '(executed by scripts/save_y72_native_plans.py --token ' + token + ')',
        observation='Eleven CPU native-evidence guards passed. CPU-tested Y72 exact seven-file version committed '
                    'on a retained branch using an isolated index; working HEAD87019/index and exact Y71/Y68 contracts '
                    'remain intact. Two resource-bound plans share one scientific program, stale goal/tampered source '
                    'refused. No native GPU steps, admission, production sync or efficacy result.', evidence=additions)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(version=ref(outputs['version']), code_commit=commit, retained_branch=BRANCH,
        working_HEAD=HEAD, main_index_unchanged=True, guard_tests_passed=11, plans=plans,
        native_executed=False, admitted=False, production_sync=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 300)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        save(args.token)
    finally:
        lock.close()
