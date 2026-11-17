"""Save the CPU-tested Y70 bundle and pending checks for either authorized server-2 GPU."""
import argparse
import ast
import copy
from datetime import datetime, timedelta
import hashlib
import json
import subprocess
import sys
from prepare_y70_query_scale_fusion import ROOT, CID, files, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import experiment_first


def git(*args):
    return subprocess.check_output(['git', '-c', 'core.autocrlf=false', *args], cwd=ROOT, timeout=120)


def save(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    c = next(c for c in work['candidates'] if c['id'] == CID)
    assert c['stage'] == 'implementation' and not c.get('code_commit') and not c.get('admission_receipt')
    result_path = ROOT / ('pipeline/analyses/y70_cpu_' + token + '.json')
    cpu = tick.load(result_path)
    for key in ('complete', 'implementation_test_passed', 'full_model_CPU_constructed',
        'configured_TRAIN_data_collated', 'released_backbone_loaded', 'native_YAML_semantically_unchanged',
        'decoder_initial_RNG_and_parent_weights_exact', 'decoder_all_stages_anchor_exact',
        'decoder_input_gradients_anchor_close', 'decoder_backward_finite',
        'decoder_fusion_gradient_finite_nonzero', 'nonzero_fusion_preserves_initial_and_changes_final_masks',
        'bypassed_fusion_negative_control_exact'):
        assert cpu[key] is True, key
    assert cpu['id'] == CID and cpu['parameters'] == 164695173 and cpu['additional_parameters'] == 65536
    assert cpu['decoder_fusion_calls'] == 6
    assert cpu['decoder_consumption_stage_calls'] == list(range(6)) and cpu['decoder_indicator_calls'] == 7
    assert len(cpu['component_controls']) == 2
    assert {row['dtype'] for row in cpu['component_controls']} == {'torch.float32', 'torch.float64'}
    assert all(v is True for row in cpu['component_controls'] for k, v in row.items() if k != 'dtype')
    for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward', 'native_sparse_forward', 'validation_used', 'admission'):
        assert cpu[key] is False
    cpu_program = ROOT / ('pipeline/checks/y70_cpu_' + token + '_program.py')
    payload = ast.literal_eval(ast.parse(cpu_program.read_text(encoding='utf-8')).body[0].value)
    assert payload['goal_sha256'] == cpu['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    paths = [p.relative_to(ROOT).as_posix() for p in files()]
    for path in files():
        value = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert ref(path)['sha256'] == value['sha256']
    base = 'pipeline/checks/y70_native_' + token
    out = {key: ROOT / (base + suffix) for key, suffix in (
        ('inputs', '_inputs.json'), ('version', '_version.json'), ('tests', '_guards.log'), ('cpu_version', '_cpu_version.json'))}
    destinations = [ROOT / (base + '_gpu' + str(gpu) + suffix) for gpu in (0, 1) for suffix in ('_plan.json', '_program.py')]
    assert not any(p.exists() for p in list(out.values()) + destinations)
    body = ROOT / 'scripts/probe_y70_native_smoke.py'
    ast.parse(body.read_text(encoding='utf-8'), feature_version=(3, 8))
    tested = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_y70_native_smoke_plan.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    out['tests'].write_bytes(tested.stdout + tested.stderr)
    assert tested.returncode == 0 and b'Ran 10 tests' in out['tests'].read_bytes()
    servers = monitor.load_servers(str(ROOT)); assert servers['server-2']['allowed_gpus'] == [0, 1]
    grant = ref('pipeline/authorizations/server2_gpu1_20261029.json')
    assert tick.load(closure.artifact(ROOT, grant))['authorization_granted']
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    queue = monitor.load_queue(str(ROOT))
    for server, cid, pid in [('server-1', 'AdaRES-Y69', 1135633), ('server-2', 'AdaRES-Y68', 20742), ('server-2', 'AdaRES-Y67', 100811)]:
        task = next(t for t in queue['gpu_queues'][server] if t['id'] == cid)
        assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid
        closure.admission(ROOT, task)
    paused = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    authorization = experiment_first.effective(ROOT)['authorization']; closure.artifact(ROOT, authorization)
    assert not git('diff', '--cached', '--name-only').strip(), 'Preserve unrelated staged work'
    parent = git('rev-parse', 'HEAD').decode().strip()
    assert parent == '9acd73cdc1b96be003c8450a9e22057248a0f531'
    tick.save(out['cpu_version'], dict(complete=True, id=CID, parent_head=parent,
        exact_files=[ref(p) for p in files()], cpu_test=ref(result_path), uncommitted=True,
        native_executed=False, admitted=False, production_sync=False))
    tick.save(out['inputs'], dict(complete=True, id=CID, parent_commit=parent,
        exact_files=[ref(p) for p in files()], CPU_manifest=ref(out['cpu_version']), cpu_test=ref(result_path),
        queue_before=queue_before, running_before=running_before, authorization=authorization,
        native_executed=False, admitted=False))
    git('add', '--', *paths)
    assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == set(paths)
    for path in paths: assert hashlib.sha256(git('show', ':' + path)).hexdigest() == ref(path)['sha256']
    git('commit', '--only', '-m', 'Add Y70 query-conditioned coarse and fine scale fusion', '--', *paths)
    commit = git('rev-parse', 'HEAD').decode().strip()
    assert git('rev-parse', commit + '^').decode().strip() == parent
    assert set(git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).decode().splitlines()) == set(paths)
    assert not git('diff', '--cached', '--name-only').strip()
    checked = dict(c, code_commit=commit)
    plans = {}; programs = []
    for gpu, power in ((0, 280.0), (1, 390.0)):
        frozen = dict(variant=payload['variant'], code_commit=commit, goal_sha256=payload['goal_sha256'],
            train_source_lf_sha256=payload['train_source_lf_sha256'], runtime_physical_gpu=gpu,
            runtime_power_limit_watts=power, resource_authorization=grant)
        program = ROOT / (base + '_gpu' + str(gpu) + '_program.py')
        plan_path = ROOT / (base + '_gpu' + str(gpu) + '_plan.json')
        plan = dict(complete=True, id=CID, config=c['config'], profile='experiment_first',
            code_commit=commit, goal_sha256=frozen['goal_sha256'], server='server-2', gpu=gpu,
            runtime_power_limit_watts=power, resource_authorization=grant,
            required_resource='Fresh exclusive authorized idle server-2 GPU' + str(gpu) +
                '; canonical terminal archive/empty reservation; no compute PID; exact hostname/private '
                'runtime/original data; existing' + str(power) + 'W unchanged. Preserve all live Y68/Y69/Y67 and paused Y57.',
            planned_command='python -B scripts/run_experiment_first_probe.py y70_gpu' + str(gpu) + '_smoke ' +
                program.relative_to(ROOT).as_posix() + ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id ' + str(gpu),
            checks='Actual native YAML/TRAIN/tools.train two finite loss/backward/optimizer steps; six ordered fusion calls and six actual query-language attention consumers per step, seven shared confidence calls, exact original previous query/coarse/fine object identities and original Boolean mask consumption; independent scalar convex formula on first/last expressions, query0/127 and channel0/127/255; zero first-step/nonzero second-step response, finite nonzero new gradient and actual optimizer update, all three input branch gradients finite, native modes and peak memory.',
            CPU_guard_tests_passed=10, Python38_compile=True, native_executed=False,
            CPU_checks_do_not_replace_native_backward=True)
        assert validate_frozen(checked, plan, frozen) == 7
        for field in ('goal', 'source'):
            bad = copy.deepcopy(frozen)
            if field == 'goal': bad['goal_sha256'] = '0' * 64
            else: bad['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
            refused = False
            try: validate_frozen(checked, plan, bad)
            except AssertionError: refused = True
            assert refused
        program.write_text('FROZEN=' + repr(frozen) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
        ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
        plan['program'] = ref(program); tick.save(plan_path, plan)
        plans[str(gpu)] = ref(plan_path); programs.append(ref(program))
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    for tasks in queue['gpu_queues'].values():
        for task in tasks:
            if task['status'] == 'running': closure.admission(ROOT, task)
    tick.save(out['version'], dict(complete=True, id=CID, observed_at=datetime.now().astimezone().isoformat(),
        parent_commit=parent, code_commit=commit, exact_files=[ref(p) for p in files()],
        CPU_manifest=ref(out['cpu_version']), cpu_test=ref(result_path), cpu_program=ref(cpu_program),
        native_plans=plans, guard_tests=ref(out['tests']), CPU_guard_tests_passed=10,
        same_diagnostic_body_for_both_GPU_plans=True, stale_goal_refused=True, tampered_source_refused=True,
        native_executed=False, admitted=False, pushed=False, production_sync=False,
        queue_and_running_preserved=True, all_original_admissions_valid=True, Y57_paused=True,
        authorization=authorization))
    additions = [ref(p) for p in out.values()] + list(plans.values()) + programs + [ref(result_path),
        ref(result_path.with_suffix('.log')), ref(body), ref(__file__), ref('scripts/test_y70_native_smoke_plan.py')]
    c.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        required_server='server-2', required_gpu=1, resource_work_kind='native_smoke',
        code_commit=commit, source_manifest=ref(out['cpu_version']), native_smoke_plan=plans['1'],
        native_smoke_plan_server2_gpu0=plans['0'], native_smoke_plan_server2_gpu1=plans['1'],
        blocker='Server-2 GPU0 runs Y68 and GPU1 runs Y67; server-1 GPU0 runs Y69. Preserve all three live runs and paused Y57. Native sparse two-step check waits.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Canonical terminal archive and fresh unreserved physically idle authorized GPU; select its exact bound plan before testing.',
        next_action='Validate current goal/HEAD/frozen seven files. Prefer active server-2 GPU1 plan after original Y67 fixed30 terminal verification and canonical archive, then fresh physical idle/unreserved check. If GPU0 frees first activate its exact stored GPU0 plan before execution. Actual native two-step TRAIN PASS is mandatory, followed by original full evaluation freeze/admission/gated dispatch in the same tick. No CPU-based admission, protected-run interference or replay of completed preparation. Server-1 requires its separate verified runtime/path adaptation.')
    c['evidence'].extend(additions)
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B scripts/save_y70_native_plans.py --token ' + token,
        observation='CPU-tested seven-file Y70 query-scale-fusion version saved;10 native evidence guards and stale-goal/'
            'source controls passed. Two resource-bound plans share identical scientific body. '
            'Three live original runs preserved; no GPU task steps, admission, production sync or efficacy result.', evidence=additions)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(code_commit=commit, scoped_files=7, plans=plans, version=ref(out['version']),
        native_executed=False, admitted=False, production_sync=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--token', required=True); args = parser.parse_args()
    lease(args.token, 240)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try: save(args.token)
    finally: lock.close()
