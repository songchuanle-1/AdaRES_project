"""Observe the selected live AdaRES GPU0 run without changing bound older observers."""

import argparse
from datetime import datetime
import json
import shlex

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_gpu0_single_card_trial import REMOTE, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    parser.add_argument('--id', required=True)
    args = parser.parse_args()
    lease(args.token, 120)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        queue = monitor.load_queue(str(ROOT))
        assert args.id.startswith('AdaRES-')
        task = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == args.id)
        assert task['status'] == 'running' and task['launch_verified'] and int(task['gpu']) == 0
        program = 'TASK=' + repr(dict(pid=task['pid'], config=task['config'], log=task['log'])) + '\n' + REMOTE
        observed = json.loads(monitor.ssh(server['host'], server['user'],
            shlex.quote(server['python']) + ' -B -c ' + shlex.quote(program), timeout=60))
        output = ROOT / ('pipeline/checks/ef_gpu0_runtime_' + args.token + '.json')
        assert not output.exists()
        result = dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
            id=task['id'], pid=task['pid'], observation=observed,
            assessment=evaluate(observed, task, server['remote_code_root']),
            remote_mutations=False, reused_observer=ref(ROOT / 'scripts/check_gpu0_single_card_trial.py'),
            limitation='Short runtime samples, not long-term hardware stability or trained efficacy.')
        tick.save(output, result)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == task['id'])
        candidate['evidence'].extend([ref(output), ref(__file__)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(evidence=ref(output), observed_at=result['observed_at'], assessment=result['assessment'],
            samples=[dict(progress=s['progress'], device=s['device']) for s in observed['samples']]), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
