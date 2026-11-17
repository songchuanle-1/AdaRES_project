"""Archive Y56 only after exact fixed-30 checkpoint and process checks."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from dispatch_prepared_candidates import snapshot


REMOTE = r'''
import gc,hashlib,json,pathlib,re,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
assert not pathlib.Path('/proc/13738').exists()
log=pathlib.Path(P['log']).resolve()
log.relative_to(pathlib.Path(P['root'])/'logs')
raw=log.read_bytes()
text=raw.decode('utf-8',errors='replace')
assert 'Val[30/30]' in text and '363/363' in text[-10000:]
assert not any(v in text[-10000:] for v in ('Traceback','RuntimeError','CUDA out of memory'))
announced=re.findall(r'save last pth to ([^\s]+/last\.pth)',text)
last=pathlib.Path(announced[-1]).resolve()
last.relative_to(pathlib.Path(P['work'])/P['basename'])
assert last.parent.name=='20261021_004502'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
 return h.hexdigest()
import torch
torch.set_num_threads(2)
checkpoints=[]
for path in (last,last.with_name('best.pth')):
 stat=(path.stat().st_size,path.stat().st_mtime_ns)
 checkpoint=torch.load(str(path),map_location='cpu')
 assert all(k in checkpoint for k in ('model','optimizer','scheduler','meta'))
 epoch=int(checkpoint['meta']['epoch'])
 assert checkpoint['optimizer']['state'] and checkpoint['optimizer']['param_groups']
 assert int(checkpoint['scheduler']['last_epoch'])==epoch
 if path==last:assert epoch==30
 checkpoints.append(dict(path=str(path),epoch=epoch,
  best_miou=float(checkpoint['meta']['best_miou']),
  optimizer_state_count=len(checkpoint['optimizer']['state']),
  scheduler_last_epoch=int(checkpoint['scheduler']['last_epoch']),sha256=sha(path)))
 del checkpoint
 gc.collect()
 assert stat==(path.stat().st_size,path.stat().st_mtime_ns)
assert checkpoints[0]['best_miou']==checkpoints[1]['best_miou']
assert not pathlib.Path('/proc/13738').exists()
metrics=re.findall(r'^mIOU: ([\d.]+)\. Acc_50: ([\d.]+)\. Acc_25: ([\d.]+)',text,re.M)
print(json.dumps(dict(complete=True,id='AdaRES-Y56',process_absent=True,
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 log=str(log),log_sha256=hashlib.sha256(raw).hexdigest(),checkpoints=checkpoints,
 final_logged_metrics=metrics[-1] if metrics else None,CUDA_initialized=torch.cuda.is_initialized(),
 full_frozen_evaluation_executed=False,hardware_settings_changed=False)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        queue = monitor.load_queue(str(ROOT))
        tasks = queue['gpu_queues']['server-2']
        task = next(t for t in tasks if t['id'] == 'AdaRES-Y56')
        paused = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
        assert task['status'] == 'running' and task['pid'] == 13738 and int(task['gpu']) == 0
        assert paused['status'] == 'paused' and paused['dispatch_hold']
        basename = Path(task['config']).stem
        base = ROOT / ('pipeline/checks/y56_fixed30_' + args.token)
        assert not base.with_suffix('.json').exists()
        payload = dict(root=server['remote_code_root'], work=server['work_dir'],
                       log=task['log'], basename=basename)
        program = 'P=' + repr(payload) + '\n' + REMOTE
        response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            server['user'] + '@' + server['host'], 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=' +
            shlex.quote(server['site_packages']) + ' ' + shlex.quote(server['python']) + ' -S -B -'],
            input=program.encode(), capture_output=True, timeout=200)
        base.with_suffix('.log').write_bytes(response.stdout + response.stderr)
        assert response.returncode == 0, 'Checkpoint verification failed; raw evidence retained'
        proof = json.loads(response.stdout)
        assert proof['complete'] and not proof['CUDA_initialized']
        live = snapshot(server, [], [])
        assert live['gpu_query_rc'] == live['compute_query_rc'] == 0
        assert 'GPU-e0b11432-b7aa-2006-1903-dcc739f26146' not in live['compute']
        proof.update(observed_at=datetime.now().astimezone().isoformat(), live=live,
                     command='python -B scripts/finalize_y56_fixed30.py --token ' + args.token)
        tick.save(base.with_suffix('.json'), proof)
        destination = ROOT / ('Checkpoints/server-2/' + basename + '/' + args.token + '/fixed30.log')
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        subprocess.run(['scp', '-q', server['user'] + '@' + server['host'] + ':' + task['log'],
                        str(destination)], check=True, timeout=60)
        assert ref(destination)['sha256'] == proof['log_sha256']
        lease(args.token, 90)
        original_pull = monitor.pull_log
        def verified_pull(project_dir, host, user, remote_log, cfg_name, state, srv_name=''):
            assert Path(project_dir) == ROOT and host == server['host'] and user == server['user']
            assert remote_log == task['log'] and cfg_name == basename
            assert state == 'done_complete' and srv_name == 'server-2'
            return str(destination)
        monitor.pull_log = verified_pull
        events = []
        try:
            monitor.finalize_task(str(ROOT), queue, 'server-2', server['host'], server['user'], task,
                basename, 'done_complete', proof['checkpoints'][0]['best_miou'],
                '30/30', '0', task['log'], events)
        finally:
            monitor.pull_log = original_pull
        experiment = monitor.load_experiment(str(ROOT), task['id'])
        experiment.update(terminal_evidence=ref(base.with_suffix('.json')),
                          terminal_checkpoint_metadata=proof['checkpoints'])
        monitor.save_experiment(str(ROOT), experiment)
        deathwatch = monitor.load_deathwatch(str(ROOT))
        deathwatch.get('server-2', {}).pop(basename, None)
        monitor.save_deathwatch(str(ROOT), deathwatch)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        for candidate in work['candidates']:
            if candidate['id'] not in ('AdaRES-Y56', 'AdaRES-Y58', 'AdaRES-Y59', 'AdaRES-Y60'):
                continue
            if candidate['id'] == 'AdaRES-Y56':
                candidate.update(stage='validation', scientific_status='fixed30_complete_frozen_review_pending',
                    next_action='Preserve the completed fixed30 run and original checkpoint lineage. Execute '
                    'the registered full frozen evaluation/comparison and trained costs without tuning '
                    'or restarting; a log best is not a scientific promotion.')
            else:
                assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
                candidate['stage'] = 'validation'
                for key in ('blocker','blocker_kind','required_server','retry_after','unblock_condition'):
                    candidate.pop(key, None)
            candidate['evidence'].append(ref(base.with_suffix('.json')))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(proof=ref(base.with_suffix('.json')), canonical_state=experiment['status'],
            checkpoints=proof['checkpoints'], events=events, Y57_unchanged_paused=True), indent=2))
    finally:
        writer.close()


if __name__ == '__main__':
    main()
