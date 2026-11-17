"""Bind Y73 native programs to its already-tested retained commit; preserve working HEAD."""
import argparse
import ast
import copy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from prepare_y73_initial_query_anchor import ROOT, CID, files, lease, monitor, ref, tick
from save_y72_native_plans import git
from check_ef_execution import validate_frozen
from y68_terminal_contract import build_contract
import closure

HEAD = '87019e7c2f640f99cb82a3e5b396321fc5b54d25'
COMMIT = '7da625ab750447eaad76e82309c89d887df93994'


def save(token):
    work = tick.load(ROOT / 'pipeline/research_work.json'); c = next(c for c in work['candidates'] if c['id'] == CID)
    assert c['stage'] == 'implementation' and c['code_commit'] == COMMIT
    cpu_version_path = closure.artifact(ROOT, c['source_manifest']); cpu_version = tick.load(cpu_version_path)
    bundle_path = closure.artifact(ROOT, cpu_version['source_snapshot']); bundle = tick.load(bundle_path)
    payload = bundle['CPU_tested_payload']; cpu_path = closure.artifact(ROOT, cpu_version['CPU_result']); cpu = tick.load(cpu_path)
    assert cpu['complete'] and cpu['implementation_test_passed'] and not cpu['CUDA_initialized'] and not cpu['admission']
    assert cpu['decoder_residual_calls'] == 6 and cpu['decoder_indicator_calls'] == 7
    assert cpu['goal_sha256'] == payload['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    assert git('rev-parse', c['retained_branch']).decode().strip() == COMMIT
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    queue = monitor.load_queue(str(ROOT)); servers = monitor.load_servers(str(ROOT))
    for server, cid, pid in [('server-2', 'AdaRES-Y68', 20742), ('server-2', 'AdaRES-Y70', 52561), ('server-1', 'AdaRES-Y69', 1135633)]:
        task = next(t for t in queue['gpu_queues'][server] if t['id'] == cid)
        assert task['status'] == 'running' and task['launch_verified'] is True and task['pid'] == pid
        closure.admission(ROOT, task)
        if cid == 'AdaRES-Y68': build_contract(task, servers[server])
    paused = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    for candidate in (next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y71'),
                      next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y72')):
        plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
        frozen = ast.literal_eval(ast.parse(closure.artifact(ROOT, plan['program']).read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(candidate, plan, frozen) == 7
    base = 'pipeline/checks/y73_native_' + token
    outputs = {key: ROOT / (base + suffix) for key, suffix in (
        ('inputs', '_inputs.json'), ('tests', '_guards.log'), ('version', '_version.json'))}
    destinations = [ROOT / (base + '_gpu' + str(g) + suffix) for g in (0, 1) for suffix in ('_plan.json', '_program.py')]
    assert not any(p.exists() for p in list(outputs.values()) + destinations)
    body = ROOT / 'scripts/probe_y73_native_smoke.py'; ast.parse(body.read_text(encoding='utf-8'), feature_version=(3, 8))
    tested = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_y73_native_smoke_plan.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    outputs['tests'].write_bytes(tested.stdout + tested.stderr)
    assert tested.returncode == 0 and b'Ran 11 tests' in outputs['tests'].read_bytes()
    grant = ref('pipeline/authorizations/server2_gpu1_20261029.json')
    assert tick.load(closure.artifact(ROOT, grant))['authorization_granted']
    tick.save(outputs['inputs'], dict(complete=True, id=CID, code_commit=COMMIT, CPU_version=ref(cpu_version_path),
        source_snapshot=ref(bundle_path), CPU_result=ref(cpu_path), native_body=ref(body),
        prior_tick_attempt=work.get('tick_attempt'), queue_before=queue_before, running_before=running_before))
    plans = {}; programs = []
    for gpu, power in ((0, 280.0), (1, 390.0)):
        frozen = dict(variant=payload['variant'], code_commit=COMMIT, goal_sha256=payload['goal_sha256'],
            train_source_lf_sha256=payload['train_source_lf_sha256'], runtime_physical_gpu=gpu,
            runtime_power_limit_watts=power, resource_authorization=grant)
        program = ROOT / (base + '_gpu' + str(gpu) + '_program.py')
        plan_path = ROOT / (base + '_gpu' + str(gpu) + '_plan.json')
        plan = dict(complete=True, id=CID, config=c['config'], profile='experiment_first', code_commit=COMMIT,
            goal_sha256=frozen['goal_sha256'], server='server-2', gpu=gpu, runtime_power_limit_watts=power,
            resource_authorization=grant, working_HEAD_preserved=HEAD, retained_branch=c['retained_branch'],
            required_resource='Fresh exclusive idle/unreserved authorized server-2 GPU' + str(gpu) +
                '; canonical terminal archive/no compute PID/exact host/private runtime/original data/current power. '
                'Preserve all live original tasks, Y57 pause and nearer Y71/Y72 native work.',
            planned_command='python -B scripts/run_experiment_first_probe.py y73_gpu' + str(gpu) + '_smoke ' +
                program.relative_to(ROOT).as_posix() + ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id ' + str(gpu),
            checks='Original native YAML/TRAIN/tools.train two finite loss/backward/optimizer steps; six current '
                'qqa_ffn identities and same undetached initial query, six actual prediction consumers/static mask '
                'features/padding identities, seven shared indicators, first/last expressions/query0/127/channel0/127/255 '
                'independent formula, zero first-step/next-step response, finite new gradient/actual update and '
                'current/initial query gradients, native modes/peak memory.',
            guard_tests_passed=11, Python38_compile=True, native_executed=False, admitted=False, native_PASS_required=True)
        assert validate_frozen(c, plan, frozen) == 7
        for field in ('goal', 'source'):
            bad = copy.deepcopy(frozen)
            if field == 'goal': bad['goal_sha256'] = '0' * 64
            else: bad['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
            rejected = False
            try: validate_frozen(c, plan, bad)
            except AssertionError: rejected = True
            assert rejected
        program.write_text('FROZEN=' + repr(frozen) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
        ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8)); plan['program'] = ref(program)
        tick.save(plan_path, plan); plans[str(gpu)] = ref(plan_path); programs.append(ref(program))
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    tick.save(outputs['version'], dict(complete=True, id=CID, code_commit=COMMIT, retained_branch=c['retained_branch'],
        working_HEAD=HEAD, exact_files=[ref(p) for p in files()], CPU_version=ref(cpu_version_path),
        native_plans=plans, guard_tests=ref(outputs['tests']), guard_tests_passed=11, same_scientific_body=True,
        stale_goal_refused=True, tampered_source_refused=True, original_admissions_valid=True,
        Y71_and_Y72_sources_and_plans_verified=True, Y68_terminal_local_contract_preserved=True,
        queue_and_running_preserved=True, Y57_paused=True, native_executed=False, admitted=False, production_sync=False,
        evidence=[ref(__file__), ref(body), ref('scripts/test_y73_native_smoke_plan.py')]))
    additions = [ref(p) for p in outputs.values()] + list(plans.values()) + programs + [ref(__file__), ref(body), ref('scripts/test_y73_native_smoke_plan.py')]
    c.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        required_server='server-2', required_gpu=0, resource_work_kind='native_smoke',
        native_smoke_plan=plans['0'], native_smoke_plan_server2_gpu0=plans['0'], native_smoke_plan_server2_gpu1=plans['1'],
        native_preparation=ref(outputs['version']),
        blocker='Server-2 GPU0 still runs original Y68 and GPU1 original Y70; server-1 GPU0 original Y69. '
                'Preserve all runs and paused Y57; no exclusive whole GPU for native task check.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Canonical terminal archive and fresh exclusive physically idle/unreserved authorized GPU; select exact matching plan.',
        next_action='Do not replay CPU/version/plan preparation. Prioritize exact Y71 guarded handoff and then '
                    'Y72. Preserve working HEAD87019, retained branch and all live source/observer/deployment '
                    'contracts. Only fresh exclusive idle/unreserved authorized GPU after canonical terminal '
                    'archive permits the exact matching Y73 native YAML/TRAIN two-step program. Actual finite '
                    'native task loss/backward/optimizer and intended-path PASS is required before original '
                    'full evaluation/cost freeze/admission, production commit selection with safe contracts, '
                    'gated dispatch and increasing native training steps. No CPU-based admission or protected '
                    'restart; server-1 needs separate verified runtime/path adaptation; Y57 stays paused.')
    c['evidence'].extend(additions)
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B -m unittest discover -s scripts -p test_y73_native_smoke_plan.py -v '
                '(executed by scripts/save_y73_native_plans.py --token ' + token + ')',
        observation='Eleven CPU native-evidence guards passed; exact retained-commit Y73 dual-card native programs '
                    'prepared, identical scientific body and stale-goal/tampered-source refusal verified. '
                    'Y71/Y72 original contracts and current HEAD87019/runs retained. No GPU task steps, '
                    'admission, production sync or efficacy result.', evidence=additions)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(version=ref(outputs['version']), code_commit=COMMIT, plans=plans,
                         guard_tests_passed=11, native_executed=False, admitted=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--token', required=True); args = parser.parse_args()
    lease(args.token, 240)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try: save(args.token)
    finally: lock.close()
