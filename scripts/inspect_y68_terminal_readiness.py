"""Read exact Y68 completion readiness; full proof remains the guarded finalizer's job."""
import argparse
import inspect
import json
from datetime import datetime
import re
import subprocess
from pathlib import Path
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from observe_y70_server2_gpu1_phase import parse_phase, phase_key
from y68_terminal_contract import build_contract, validate_identity
import closure

REMOTE = r'''
import hashlib,json,pathlib,subprocess,yaml
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path(P['root']);task=P['task'];proc=pathlib.Path('/proc')/str(task['pid'])
live=proc.is_dir()
if live:
    try:
        assert (proc/'cwd').resolve()==root
        args=(proc/'cmdline').read_bytes().split(b'\0')
        assert task['config'].encode() in args and b'tools/train.py' in args
    except FileNotFoundError:
        assert not proc.exists();live=False
for relative,digest in P['source_files'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
assert yaml.safe_load((root/task['config']).read_text())==P['expected_config']
marker=(root/'.codex_source_commit').read_text().strip();assert marker in P['deployments']
for relative,digest in P['deployments'][marker]['source_files'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
for relative,expected in P['deployments'][marker]['configs'].items():
    assert yaml.safe_load((root/relative).read_text())==expected
log=pathlib.Path(task['log'])
with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-32000));tail=stream.read().decode(errors='replace')
assert not any(s in tail for s in ('Traceback','CUDA out of memory','RuntimeError'))
latest=parse_phase(tail)
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu,power.limit','--format=csv,noheader,nounits']).decode().strip()
assert float(gpu.split(',')[-1])==280.0
compute=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
can_finalize=not live and latest['phase']=='Val' and latest['epoch']==30 and latest['step']==363
print(json.dumps(dict(complete=True,pid=task['pid'],process_live=live,latest=latest,source_marker=marker,
    exact_source_and_YAML_verified=True,can_attempt_guarded_finalizer=can_finalize,gpu=gpu,compute=compute,
    limitation='Read-only readiness only; not fixed30 checkpoint/log proof, canonical archive, native GPU PASS or efficacy.')))
'''


def inspect_run(token, label):
    server = monitor.load_servers(str(ROOT))['server-2']; closure.validate_adares_server(ROOT, server)
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    task = next(t for t in monitor.load_queue(str(ROOT))['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y68')
    validate_identity(task); closure.admission(ROOT, task)
    contract = build_contract(task, server)
    package = ROOT / 'AdaRES/ipdn/model_paper5/adares_y68_language_mask_modulation'
    payload = dict(root=server['remote_code_root'], task=task,
        source_files={p.relative_to(ROOT / 'AdaRES').as_posix(): ref(p)['sha256'] for p in package.glob('*.py')},
        expected_config=yaml.safe_load((ROOT / 'AdaRES' / task['config']).read_bytes()),
        deployments=contract['deployments'])
    assert len(payload['source_files']) == 6
    program = 'P=' + repr(payload) + '\n' + '\n'.join(inspect.getsource(f) for f in (parse_phase, phase_key)) + '\n' + REMOTE
    output = ROOT / ('pipeline/checks/y68_terminal_readiness_' + token + '_' + label + '.json')
    assert not output.exists() and not output.with_suffix('.log').exists()
    started = datetime.now().astimezone().isoformat()
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', server['user'] + '@' + server['host'],
        'PYTHONDONTWRITEBYTECODE=1 ' + server['python'] + ' -B -'], input=program.encode(), capture_output=True, timeout=30)
    output.with_suffix('.log').write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0, 'Preserve readiness query failure; never restart'
    result = json.loads(response.stdout)
    result.update(token=token, id='AdaRES-Y68', observed_start_local=started,
                  observed_end_local=datetime.now().astimezone().isoformat(), evidence=[ref(__file__), ref(output.with_suffix('.log'))])
    tick.save(output, result)
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y68')
    candidate['evidence'].extend([ref(output), ref(__file__)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    print(json.dumps(dict(evidence=ref(output), **{k: result[k] for k in ('process_live','latest','can_attempt_guarded_finalizer','gpu')})))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--token', required=True); p.add_argument('--label', required=True); a = p.parse_args()
    assert re.fullmatch('[a-z0-9_]{1,40}', a.label)
    lease(a.token, 120)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try: inspect_run(a.token, a.label)
    finally: lock.close()
