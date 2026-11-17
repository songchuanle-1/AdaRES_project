"""Save exactly the CPU-tested Y67 files and an unexecuted native plan."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
import subprocess
import sys
from prepare_y67_stage_readout_norm import ROOT, CID, files, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import experiment_first


def git(*args):
    return subprocess.check_output(['git', '-c', 'core.autocrlf=false', *args], cwd=ROOT, timeout=120)


def save(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    c = next(c for c in work['candidates'] if c['id'] == CID)
    assert c['stage'] == 'blocked' and c['blocker_kind'] == 'exclusive_gpu_slot'
    assert not any(c.get(k) for k in ('code_commit', 'native_smoke_plan', 'admission_receipt'))
    manifest = tick.load(closure.artifact(ROOT, c['source_manifest']))
    cpu_path = closure.artifact(ROOT, manifest['cpu_test'])
    cpu = tick.load(cpu_path)
    assert manifest['uncommitted'] and manifest['exact_files'] == [ref(p) for p in files()]
    assert cpu['id'] == CID and cpu['implementation_test_passed'] and not cpu['CUDA_initialized']
    assert cpu['decoder_all_stages_anchor_exact'] and cpu['decoder_all_norm_gradients_finite_nonzero']
    cpu_program = ROOT / ('pipeline/checks/y67_cpu_' + token + '_attempt2_program.py')
    payload = ast.literal_eval(ast.parse(cpu_program.read_text(encoding='utf-8')).body[0].value)
    assert cpu['goal_sha256'] == payload['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    paths = [p.relative_to(ROOT).as_posix() for p in files()]
    for path in files():
        value = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert ref(path)['sha256'] == value['sha256']
    base = 'pipeline/checks/y67_native_' + token
    out = {key: ROOT / (base + suffix) for key, suffix in (
        ('inputs', '_inputs.json'), ('version', '_version.json'), ('plan', '_plan.json'),
        ('program', '_program.py'), ('tests', '_guards.log'))}
    assert not any(p.exists() for p in out.values())
    body = ROOT / 'scripts/probe_y67_native_smoke.py'
    ast.parse(body.read_text(encoding='utf-8'), feature_version=(3, 8))
    tested = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_y67_native_smoke_plan.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    out['tests'].write_bytes(tested.stdout + tested.stderr)
    assert tested.returncode == 0 and b'Ran 8 tests' in out['tests'].read_bytes()
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    queue = monitor.load_queue(str(ROOT))
    for server, cid, pid in [('server-1', 'AdaRES-Y66', 1973300), ('server-2', 'AdaRES-Y65', 112356)]:
        task = next(t for t in queue['gpu_queues'][server] if t['id'] == cid)
        assert task['pid'] == pid and task['status'] == 'running' and task['launch_verified']
        assert task['code_commit'] == '882d61537fe7ffcb383c8f51f12fea36d1e53a6d'
        closure.admission(ROOT, task)
    paused = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    authorization = experiment_first.effective(ROOT)['authorization']
    closure.artifact(ROOT, authorization)
    assert not git('diff', '--cached', '--name-only').strip(), 'Preserve staged user work'
    parent = git('rev-parse', 'HEAD').decode().strip()
    assert parent == manifest['parent_head'] == '882d61537fe7ffcb383c8f51f12fea36d1e53a6d'
    tick.save(out['inputs'], dict(complete=True, id=CID, parent_commit=parent,
        exact_files=[ref(p) for p in files()], CPU_manifest=c['source_manifest'], cpu_test=ref(cpu_path),
        queue_before=queue_before, running_before=running_before, authorization=authorization,
        native_executed=False, admitted=False))
    git('add', '--', *paths)
    assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == set(paths)
    for path in paths: assert hashlib.sha256(git('show', ':' + path)).hexdigest() == ref(path)['sha256']
    git('commit', '--only', '-m', 'Add Y67 independent prediction-stage normalization', '--', *paths)
    commit = git('rev-parse', 'HEAD').decode().strip()
    assert git('rev-parse', commit + '^').decode().strip() == parent
    assert set(git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).decode().splitlines()) == set(paths)
    assert not git('diff', '--cached', '--name-only').strip()
    frozen = dict(variant=payload['variant'], code_commit=commit, goal_sha256=payload['goal_sha256'],
        train_source_lf_sha256=payload['train_source_lf_sha256'])
    plan = dict(complete=True, id=CID, config=c['config'], profile='experiment_first',
        code_commit=commit, goal_sha256=frozen['goal_sha256'], server='server-2', gpu=0,
        required_resource='Fresh exclusive authorized idle server-2 GPU0; no compute PID or '
            'canonical reservation; memory<1024MiB/util<10; exact hostname/private runtime/data '
            'and approved280W. Preserve live Y65/Y66 and paused Y57/GPU1. Server-1 requires a '
            'separate verified runtime/path-adapted plan before execution.',
        planned_command='python -B scripts/run_experiment_first_probe.py y67_smoke ' +
            out['program'].relative_to(ROOT).as_posix() + ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id 0',
        checks='Native YAML/TRAIN/tools.train two finite loss/backward/optimizer steps; seven '
            'private norms once per step; independent LayerNorm rows0/64/127; exact normalized '
            'tensor consumption by native mask/score paths and mask consumption by shared indicator; '
            'all14 affine gradients finite/nonzero with actual optimizer parameter updates; '
            'unchanged training modes and peak memory.',
        Python38_compile=True, CPU_guard_tests_passed=8, native_executed=False,
        CPU_checks_do_not_replace_native_backward=True)
    checked = dict(c, code_commit=commit)
    assert validate_frozen(checked, plan, frozen) == 7
    for field in ('goal', 'source'):
        bad = copy.deepcopy(frozen)
        if field == 'goal': bad['goal_sha256'] = '0' * 64
        else: bad['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
        refused = False
        try: validate_frozen(checked, plan, bad)
        except AssertionError: refused = True
        assert refused
    out['program'].write_text('FROZEN=' + repr(frozen) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ast.parse(out['program'].read_text(encoding='utf-8'), feature_version=(3, 8))
    plan['program'] = ref(out['program'])
    tick.save(out['plan'], plan)
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    for tasks in queue['gpu_queues'].values():
        for task in tasks:
            if task['status'] == 'running': closure.admission(ROOT, task)
    tick.save(out['version'], dict(complete=True, id=CID, observed_at=datetime.now().astimezone().isoformat(),
        parent_commit=parent, code_commit=commit, exact_files=[ref(p) for p in files()],
        original_CPU_manifest=c['source_manifest'], cpu_test=ref(cpu_path), cpu_program=ref(cpu_program),
        native_plan=ref(out['plan']), guard_tests=ref(out['tests']), CPU_guard_tests_passed=8,
        stale_goal_refused=True, tampered_source_refused=True, native_executed=False,
        admitted=False, pushed=False, production_sync=False, hardware_changed=False,
        queue_and_running_preserved=True, Y65_Y66_original_receipts_valid=True,
        Y57_paused=True, authorization=authorization))
    additions = [ref(p) for p in out.values()] + [ref(body), ref(__file__), ref('scripts/test_y67_native_smoke_plan.py')]
    c.update(code_commit=commit, native_smoke_plan=ref(out['plan']),
        next_action=plan['planned_command'] + '; only actual native PASS and original frozen '
            'full post-training protocol permit admission and same-tick dispatch. Preserve '
            'code882d live Y65/Y66 and Y57 pause. Do not replay completed preparation.')
    c['evidence'].extend(additions)
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B scripts/save_y67_native_plan.py --token ' + token,
        observation='CPU-tested seven-file version saved;8 native stage/update guards and '
            'stale-goal/source controls passed. Live Y65/Y66 receipts, queue and running preserved. '
            'Native sparse two-step test waits for exclusive authorized GPU0; efficacy unknown.', evidence=additions)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(code_commit=commit, scoped_files=7, plan=ref(out['plan']),
        version=ref(out['version']), native_executed=False, admitted=False, production_sync=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--token', required=True); args = parser.parse_args()
    lease(args.token, 240)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try: save(args.token)
    finally: lock.close()
