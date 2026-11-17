"""Observe the admitted Y64 through Train/Val transitions without replaying launch."""
import argparse
import ast
from datetime import datetime
import json
import subprocess
import sys
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import closure


def parse_phase(text):
    import re
    rows = re.findall(r'(Train|Val)\[(\d+)/(\d+)\]:[^\r\n]*?(\d+)/(\d+)\s*\[', text)
    assert rows, 'No native Train or Val phase in bounded tail'
    phase, epoch, total, step, count = rows[-1]
    result = dict(phase=phase, epoch=int(epoch), total=int(total), step=int(step), count=int(count))
    phase_key(result)
    return result


def phase_key(value):
    assert value['phase'] in ('Train', 'Val')
    assert value['total'] == 30 and 1 <= value['epoch'] <= 30
    assert value['count'] == (4686 if value['phase'] == 'Train' else 363)
    assert 0 <= value['step'] <= value['count']
    return value['epoch'], int(value['phase'] == 'Val'), value['step']


def progressed(before, after):
    start, end = phase_key(before['latest']), phase_key(after['latest'])
    assert end >= start, 'Native phase/step regressed'
    assert after['log_bytes'] >= before['log_bytes'], 'Native log truncated'
    return end > start and after['log_bytes'] > before['log_bytes']


REMOTE = r'''
import datetime,hashlib,json,pathlib,subprocess,time,yaml
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
root=pathlib.Path(P['root']);task=P['task']
def sample():
    proc=pathlib.Path('/proc')/str(task['pid'])
    assert proc.is_dir() and (proc/'cwd').resolve()==root
    args=(proc/'cmdline').read_bytes().split(b'\0')
    assert task['config'].encode() in args and b'tools/train.py' in args
    log=pathlib.Path(task['log'])
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-32000));text=stream.read().decode(errors='replace')
    assert not any(v in text for v in ('Traceback','CUDA out of memory','RuntimeError'))
    return dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
                latest=parse_phase(text),log_bytes=log.stat().st_size)
before=sample();samples=[before];confirmed=False
for attempt in range(3):
    time.sleep(5);after=sample();samples.append(after)
    if progressed(before,after):
        confirmed=True
        break
for relative,digest in P['source_files'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
cfg=yaml.safe_load((root/task['config']).read_text())
assert cfg==P['expected_config']
assert cfg['train']['seed']==1999 and cfg['train']['epochs']==30
assert (root/'.codex_source_commit').read_text().strip()==task['code_commit']
uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader']).decode().strip()
owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']).decode()
assert [int(row.split(',')[1]) for row in owners.splitlines() if row.split(',')[0].strip()==uuid]==[int(task['pid'])]
print(json.dumps(dict(complete=True,before=before,after=after,samples=samples,
    native_progress_confirmed=confirmed,process_identity_verified=True,pid=task['pid'],log=task['log'],
    source_files_verified=len(P['source_files']),native_YAML_semantics_verified=True,
    source_marker=task['code_commit'],compute_owners=owners,authorized_gpu_uuid=uuid,
    gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.limit','--format=csv,noheader,nounits']).decode().strip(),
    no_bounded_tail_error=True,training_efficacy='Unknown; full frozen evaluation not executed.',
    limitation='A live process without advancing native Train/Val counters is unconfirmed, never a restart authorization.')))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 150)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-1']
        closure.validate_adares_server(ROOT, server)
        queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
        task = next(t for t in monitor.load_queue(str(ROOT))['gpu_queues']['server-1'] if t['id'] == 'AdaRES-Y64')
        assert task['status'] == 'running' and task['launch_verified'] and str(task['gpu']) == '0'
        assert monitor.load_running(str(ROOT))['server-1']['0'] == task['id']
        closure.admission(ROOT, task)
        cfg = yaml.safe_load((ROOT / 'AdaRES' / task['config']).read_text(encoding='utf-8'))
        def adapt(value):
            if isinstance(value, dict): return {k: adapt(v) for k, v in value.items()}
            if isinstance(value, list): return [adapt(v) for v in value]
            if isinstance(value, str):
                for mapping in server['runtime_path_mappings']:
                    old, new = mapping['source'].rstrip('/'), mapping['target'].rstrip('/')
                    if value == old or value.startswith(old + '/'): value = new + value[len(old):]
            return value
        folder = ROOT / 'AdaRES/ipdn/model_paper5/adares_y64_directed_containment'
        files = {p.relative_to(ROOT / 'AdaRES').as_posix(): ref(p)['sha256'] for p in folder.glob('*.py')}
        assert len(files) == 6
        payload = dict(root=server['remote_code_root'], task=task, source_files=files, expected_config=adapt(cfg))
        this = Path(__file__).read_text(encoding='utf-8')
        tree = ast.parse(this)
        functions = '\n\n'.join(ast.get_source_segment(this, node) for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in ('parse_phase', 'phase_key', 'progressed'))
        program = 'P=' + repr(payload) + '\n' + functions + '\n' + REMOTE
        ast.parse(program, feature_version=(3, 8))
        output = ROOT / ('pipeline/checks/y64_server1_phase_progress_' + args.token + '.json')
        assert not output.exists() and not output.with_suffix('.log').exists()
        started = datetime.now().astimezone().isoformat()
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            server['user'] + '@' + server['host'], 'PYTHONDONTWRITEBYTECODE=1 ' + server['python'] + ' -B -'],
            input=program.encode(), capture_output=True, timeout=45)
        output.with_suffix('.log').write_bytes(result.stdout + result.stderr)
        assert result.returncode == 0, 'Preserve observer failure; never relaunch'
        observation = json.loads(result.stdout)
        assert observation['native_progress_confirmed'] == progressed(observation['before'], observation['after'])
        observation.update(id=task['id'], token=args.token, observed_start_local=started,
            observed_end_local=datetime.now().astimezone().isoformat(),
            evidence=[ref(__file__), ref(output.with_suffix('.log'))])
        tick.save(output, observation)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == task['id'])
        assert candidate['stage'] == 'training'
        candidate['evidence'].extend([ref(output), ref(__file__)])
        candidate['latest_training_observation'] = ref(output)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
        print(json.dumps(dict(evidence=ref(output), before=observation['before'], after=observation['after'],
            native_progress_confirmed=observation['native_progress_confirmed'], gpu=observation['gpu'])))
    finally:
        lock.close()


from pathlib import Path

if __name__ == '__main__':
    main()
