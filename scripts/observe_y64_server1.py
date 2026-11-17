"""Read-only observation of the admitted Y64; no archive or dispatch operation."""
import argparse
import ast
from datetime import datetime
import json
import subprocess
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from capture_server1_training import BODY
import closure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 120)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-1']
        assert server['host'] == '192.168.1.175' and server['allowed_gpus'] == [0]
        queue = monitor.load_queue(str(ROOT))
        task = next(t for t in queue['gpu_queues']['server-1'] if t['id'] == 'AdaRES-Y64')
        assert task['status'] == 'running' and task['launch_verified'] and str(task['gpu']) == '0'
        assert monitor.load_running(str(ROOT))['server-1']['0'] == task['id']
        closure.admission(ROOT, task)
        cfg = yaml.safe_load((ROOT / 'AdaRES' / task['config']).read_text(encoding='utf-8'))
        def adapt(v):
            if isinstance(v, dict): return {k: adapt(x) for k, x in v.items()}
            if isinstance(v, list): return [adapt(x) for x in v]
            if isinstance(v, str):
                for m in server['runtime_path_mappings']:
                    old, new = m['source'].rstrip('/'), m['target'].rstrip('/')
                    if v == old or v.startswith(old + '/'): v = new + v[len(old):]
            return v
        folder = ROOT / 'AdaRES/ipdn/model_paper5/adares_y64_directed_containment'
        files = {p.relative_to(ROOT / 'AdaRES').as_posix(): ref(p)['sha256'] for p in folder.glob('*.py')}
        assert len(files) == 6
        payload = dict(root=server['remote_code_root'], task=task, source_files=files, expected_config=adapt(cfg))
        body = BODY.replace("training_efficacy='Unknown; no validation result yet.'",
                            "training_efficacy='Unknown; full frozen evaluation not executed.'")
        body = body.replace('print(json.dumps(dict(complete=True',
            "uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader']).decode().strip()\n"
            "owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']).decode()\n"
            "assert [int(row.split(',')[1]) for row in owners.splitlines() if row.split(',')[0].strip()==uuid]==[int(task['pid'])]\n"
            "assert (root/'.codex_source_commit').read_text().strip()==task['code_commit']\n"
            "print(json.dumps(dict(complete=True,compute_owners=owners,authorized_gpu_uuid=uuid")
        program = 'P=' + repr(payload) + '\n' + body
        ast.parse(program, feature_version=(3, 8))
        output = ROOT / ('pipeline/checks/y64_server1_progress_' + args.token + '.json')
        assert not output.exists() and not output.with_suffix('.log').exists()
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            server['user'] + '@' + server['host'], 'PYTHONDONTWRITEBYTECODE=1 ' + server['python'] + ' -B -'],
            input=program.encode(), capture_output=True, timeout=45)
        output.with_suffix('.log').write_bytes(result.stdout + result.stderr)
        assert result.returncode == 0, 'Preserve observation failure, never relaunch'
        observation = json.loads(result.stdout)
        observation.update(id=task['id'], token=args.token, observed_at=datetime.now().astimezone().isoformat(),
            evidence=[ref(__file__), ref('scripts/capture_server1_training.py'), ref(output.with_suffix('.log'))])
        tick.save(output, observation)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == task['id'])
        assert candidate['stage'] == 'training'
        candidate['evidence'].append(ref(output))
        candidate['latest_training_observation'] = ref(output)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(evidence=ref(output), before=observation['before'], after=observation['after'], gpu=observation['gpu'])))
    finally:
        lock.close()


if __name__ == '__main__': main()
