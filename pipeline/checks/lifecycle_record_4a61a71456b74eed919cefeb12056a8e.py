"""Record actual lifecycle regression execution under this foreground lease."""

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_y60_stage_heads import lease, monitor, ref, tick

TOKEN = '4a61a71456b74eed919cefeb12056a8e'
lease(TOKEN, 90)
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    base = ROOT / ('pipeline/checks/gpu0_lifecycle_tests_' + TOKEN)
    assert not base.with_suffix('.json').exists() and not base.with_suffix('.log').exists()
    command = [sys.executable, '-B', '-m', 'unittest', 'test_gpu0_lifecycle_v2']
    result = subprocess.run(command, cwd=ROOT / 'scripts', capture_output=True, timeout=45)
    base.with_suffix('.log').write_bytes(result.stdout + result.stderr)
    assert result.returncode == 0 and b'Ran 10 tests' in result.stderr and b'OK' in result.stderr
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    assert head == '6ac772a0c3eb4b5e598d25ce00c0ab7f203ce1e3'
    queue = monitor.load_queue(str(ROOT))['gpu_queues']['server-2']
    assert next(t for t in queue if t['id'] == 'AdaRES-Y57')['status'] == 'paused'
    running = [t for t in queue if t['status'] in ('running', 'claiming')]
    assert len(running) == 1 and running[0]['id'] == 'AdaRES-Y59' and int(running[0]['gpu']) == 0
    files = [ROOT / 'scripts' / name for name in (
        'recheck_gpu0_lifecycle.py', 'test_gpu0_lifecycle.py',
        'recheck_gpu0_lifecycle_v2.py', 'test_gpu0_lifecycle_v2.py')]
    tick.save(base.with_suffix('.json'), dict(complete=True,
        observed_at=datetime.now().astimezone().isoformat(), command=command,
        returncode=result.returncode, tests=10, passed=True, log=ref(base.with_suffix('.log')),
        sources=[ref(path) for path in files],
        executed_resource_reconciliation=ref('pipeline/checks/gpu0_lifecycle_' + TOKEN + '.json'),
        limitation='CPU lifecycle regressions and fresh recorded resources only. The v2 idle-slot '
                   'transition is unit-tested, not an actual idle-GPU observation or native smoke.',
        git_head=head, model_changed=False, native_smoke_executed=False,
        admission_or_launch_performed=False, runtime_hardware_changed=False))
    decision = monitor.log_decision(str(ROOT), 'other',
        '资源复查错误地将已完成Y58的冻结后评待办写成训练前准入，如何修复而不改历史证据？',
        '保留旧脚本和本轮原始输出；分离Y56/Y58后评、Y60原生smoke和Y61版本准备。后续资源复查用recheck_gpu0_lifecycle_v2.py。',
        '当前GPU0仅Y59实测训练增长，280W/84C且热降频；GPU1不获准。10项CPU回归通过，不证明GPU空闲或模型有效。',
        str(base.with_suffix('.json').relative_to(ROOT)))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    additions = [ref(base.with_suffix('.json')), ref(base.with_suffix('.log')), ref(__file__),
                 ref(decision)] + [ref(path) for path in files]
    for candidate in work['candidates']:
        if candidate['id'] in ('AdaRES-Y56', 'AdaRES-Y58', 'AdaRES-Y60', 'AdaRES-Y61'):
            candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(base.with_suffix('.json')), tests=10,
                          decision=ref(decision), model_changed=False), indent=2))
finally:
    lock.close()
