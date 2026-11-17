"""Persist handoff and actual bookkeeping regression results under the current lease."""

from datetime import datetime
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_y60_stage_heads import lease, monitor, ref, tick
import closure

TOKEN = '3e61cded0e934db5bb4f9abd4451c12c'
lease(TOKEN, 120)
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    base = ROOT / ('pipeline/checks/y60_handoff_' + TOKEN)
    assert not base.with_suffix('.json').exists() and not base.with_suffix('.log').exists()
    command = [sys.executable, '-B', '-m', 'unittest', 'test_gpu0_lifecycle_v3']
    tests = subprocess.run(command, cwd=ROOT / 'scripts', capture_output=True, timeout=45)
    base.with_suffix('.log').write_bytes(tests.stdout + tests.stderr)
    assert tests.returncode == 0 and b'Ran 8 tests' in tests.stderr and b'OK' in tests.stderr
    queue = monitor.load_queue(str(ROOT))['gpu_queues']['server-2']
    paused = next(t for t in queue if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold']
    active = [t for t in queue if t['status'] in ('running', 'claiming', 'waiting')]
    assert len(active) == 1 and active[0]['id'] == 'AdaRES-Y60'
    task = active[0]
    assert task['status'] == 'running' and int(task['gpu']) == 0 and task['launch_verified']
    assert task['fixed_epochs'] == task['min_epochs'] == task['patience'] == 30
    closure.admission(ROOT, task)
    assert monitor.load_experiment(str(ROOT), 'AdaRES-Y59')['status'] == 'done_complete'
    runtime_path = ROOT / ('pipeline/checks/ef_gpu0_runtime_' + TOKEN + '.json')
    runtime = tick.load(runtime_path)
    assert runtime['id'] == 'AdaRES-Y60' and runtime['pid'] == task['pid']
    assert all(runtime['assessment'][k] for k in ('process_identity_verified', 'native_progress_verified',
                                                'power_cap_preserved', 'gpu0_compute_owner_verified'))
    resource_path = ROOT / ('pipeline/checks/gpu0_lifecycle_v3_' + TOKEN + '.json')
    resource = tick.load(resource_path)
    assert not resource['idle_authorized_gpus']
    observed = resource['live']['tasks']['AdaRES-Y60']
    assert observed['process_live'] and task['config'] in observed['cmdline']
    steps = closure.steps(observed['tail'])
    assert len(set(steps)) >= 2
    failure = ROOT / ('pipeline/checks/lifecycle_idle_failure_' + TOKEN + '.json')
    sources = [ROOT / 'scripts' / name for name in (
        'recheck_gpu0_lifecycle.py', 'recheck_gpu0_lifecycle_v2.py',
        'recheck_gpu0_lifecycle_v3.py', 'test_gpu0_lifecycle_v3.py')]
    tick.save(base.with_suffix('.json'), dict(complete=True,
        observed_at=datetime.now().astimezone().isoformat(), id='AdaRES-Y60', pid=task['pid'],
        runtime_evidence=ref(runtime_path), latest_resource_evidence=ref(resource_path),
        latest_native_steps=[steps[0], steps[-1]], hardware=resource['hardware'],
        actual_test_command=command, test_returncode=tests.returncode, regression_tests_passed=8,
        test_log=ref(base.with_suffix('.log')), preserved_operational_failure=ref(failure),
        source_evidence=[ref(path) for path in sources],
        old_collectors_preserved=True, model_changed=False, new_git_commit=False,
        no_GPU1_or_scheduler_change=True, scientific_efficacy='Unknown; full frozen evaluation pending.'))
    decision = monitor.log_decision(str(ROOT), 'other',
        '空卡采集器已移除retry_after后旧生命周期包装器报KeyError，如何防止记录错误阻断实际接续？',
        '保留旧v1/v2和失败；在真实空闲GPU0守卫下完成Y60两步smoke、准入/门控派发和递增步数验证。新v3按生命周期与可选retry判断，不以Y60必须待检作固定前提。',
        '8项CPU回归及v3当前真实忙卡复查通过；Y59 last30/best13已核验，Y60真实运行，Y56/Y58/Y59仅冻结后评，Y61只版本/原生检查待办；不改变模型、调度或资源许可。',
        ref(base.with_suffix('.json'))['path'])
    work = tick.load(ROOT / 'pipeline/research_work.json')
    additions = [ref(base.with_suffix('.json')), ref(base.with_suffix('.log')), ref(__file__),
                 ref(failure), ref(decision)] + [ref(path) for path in sources]
    for candidate in work['candidates']:
        if candidate['id'] in ('AdaRES-Y56', 'AdaRES-Y58', 'AdaRES-Y59', 'AdaRES-Y60', 'AdaRES-Y61'):
            candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
    attempt = work['tick_attempt']
    assert attempt['candidate_id'] == 'AdaRES-Y60' and attempt['kind'] == 'native_smoke'
    candidate = next(c for c in work['candidates'] if c['id'] == attempt['candidate_id'])
    assert all(e in candidate['evidence'] for e in attempt['evidence'])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(base.with_suffix('.json')), decision=ref(decision),
        latest_native_steps=[steps[0], steps[-1]], hardware=resource['hardware'], tests_passed=8), indent=2))
finally:
    lock.close()
