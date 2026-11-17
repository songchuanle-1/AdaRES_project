"""Reconcile completed Y58 and the already launched Y59 with exact runtime evidence."""

import argparse
import ast
from datetime import datetime, timedelta
import json
from pathlib import Path
import shlex
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from dispatch_prepared_candidates import snapshot
from check_ef_execution import validate_frozen
import closure


REMOTE = r'''
import gc,hashlib,json,math,pathlib,re,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
assert not pathlib.Path('/proc/'+str(P['pid'])).exists()
processes=subprocess.check_output(['ps','-u','songcl','-o','pid=,args=']).decode()
assert not any('tools/train.py' in line and P['config'] in line for line in processes.splitlines())
log=pathlib.Path(P['log']).resolve()
log.relative_to(pathlib.Path(P['root'])/'logs')
raw=log.read_bytes()
text=raw.decode('utf-8',errors='replace')
assert 'Val[30/30]' in text and '363/363' in text[-10000:]
assert any('Train[30/30]' in line and '4686/4686' in line for line in text.splitlines())
assert not any(value in text[-10000:] for value in ('Traceback','RuntimeError','CUDA out of memory'))
announced=re.findall(r'save last pth to ([^\s]+/last\.pth)',text)
assert announced
last=pathlib.Path(announced[-1]).resolve()
last.relative_to(pathlib.Path(P['work'])/P['basename'])
assert last.parent.name==P['checkpoint_directory']

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
 return h.hexdigest()

import torch
torch.set_num_threads(2)
assert not torch.cuda.is_initialized()
checkpoints=[]
for path in (last,last.with_name('best.pth')):
 stat=(path.stat().st_size,path.stat().st_mtime_ns)
 checkpoint=torch.load(str(path),map_location='cpu')
 assert all(key in checkpoint for key in ('model','optimizer','scheduler','meta'))
 assert checkpoint['model'] and checkpoint['optimizer']['state'] and checkpoint['optimizer']['param_groups']
 epoch=int(checkpoint['meta']['epoch'])
 assert int(checkpoint['scheduler']['last_epoch'])==epoch
 if path==last:assert epoch==30
 best=float(checkpoint['meta']['best_miou'])
 assert math.isfinite(best)
 checkpoints.append(dict(path=str(path),epoch=epoch,best_miou=best,
  optimizer_state_count=len(checkpoint['optimizer']['state']),
  scheduler_last_epoch=int(checkpoint['scheduler']['last_epoch']),sha256=sha(path)))
 del checkpoint
 gc.collect()
 assert stat==(path.stat().st_size,path.stat().st_mtime_ns)
assert checkpoints[0]['best_miou']==checkpoints[1]['best_miou']
assert not pathlib.Path('/proc/'+str(P['pid'])).exists()
metrics=re.findall(r'^mIOU: ([\d.]+)\. Acc_50: ([\d.]+)\. Acc_25: ([\d.]+)',text,re.M)
print(json.dumps(dict(complete=True,id=P['id'],process_absent=True,
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 log=str(log),log_sha256=hashlib.sha256(raw).hexdigest(),checkpoints=checkpoints,
 final_logged_metrics=metrics[-1] if metrics else None,CUDA_initialized=torch.cuda.is_initialized(),
 full_frozen_evaluation_executed=False,hardware_settings_changed=False)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 360)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        queue = monitor.load_queue(str(ROOT))
        tasks = {t['id']: t for t in queue['gpu_queues']['server-2']}
        old, current = tasks['AdaRES-Y58'], tasks['AdaRES-Y59']
        assert old['status'] == 'running' and old['pid'] == 77094 and int(old['gpu']) == 0
        assert current['status'] == 'running' and current['launch_verified'] and int(current['gpu']) == 0
        assert tasks['AdaRES-Y57']['status'] == 'paused' and tasks['AdaRES-Y57']['dispatch_hold']
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidates = {c['id']: c for c in work['candidates'] if c['id'] in ('AdaRES-Y58', 'AdaRES-Y59')}
        identities = []
        for task in (old, current):
            closure.admission(ROOT, task)
            candidate = candidates[task['id']]
            plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
            program = closure.artifact(ROOT, plan['program'])
            frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
            assert validate_frozen(candidate, plan, frozen) == 7
            identities.append(dict(id=task['id'], exact_files=7, admission_verified=True))
        runtime_path = ROOT / ('pipeline/checks/ef_gpu0_runtime_' + args.token + '.json')
        runtime = tick.load(runtime_path)
        assert runtime['id'] == current['id'] and runtime['pid'] == current['pid']
        age = (datetime.now().astimezone() - datetime.fromisoformat(runtime['observed_at'])).total_seconds()
        assert 0 <= age < 1200
        assert all(runtime['assessment'][key] for key in
                   ('process_identity_verified', 'native_progress_verified',
                    'power_cap_preserved', 'gpu0_compute_owner_verified'))
        basename = Path(old['config']).stem
        base = ROOT / ('pipeline/checks/y58_fixed30_handoff_' + args.token)
        assert not base.with_suffix('.json').exists() and not base.with_suffix('.log').exists()
        payload = dict(id=old['id'], pid=old['pid'], root=server['remote_code_root'],
                       work=server['work_dir'], log=old['log'], config=old['config'],
                       basename=basename, checkpoint_directory='20261021_130636')
        response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            server['user'] + '@' + server['host'],
            'PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=' +
            shlex.quote(server['site_packages']) + ' ' + shlex.quote(server['python']) + ' -S -B -'],
            input=('P=' + repr(payload) + '\n' + REMOTE).encode(), capture_output=True, timeout=200)
        base.with_suffix('.log').write_bytes(response.stdout + response.stderr)
        assert response.returncode == 0, 'Checkpoint verification failed; preserve raw evidence'
        proof = json.loads(response.stdout)
        assert proof['complete'] and not proof['CUDA_initialized']
        live = snapshot(server, [old, current], [])
        assert live['gpu_query_rc'] == live['compute_query_rc'] == 0
        assert not live['tasks'][old['id']]['process_live']
        now_current = live['tasks'][current['id']]
        assert now_current['process_live'] and now_current['cwd'] == server['remote_code_root']
        assert current['config'] in now_current['cmdline'] and 'tools/train.py' in now_current['cmdline']
        assert len(set(closure.steps(now_current['tail']))) >= 2
        observed_at = datetime.now().astimezone()
        proof.update(observed_at=observed_at.isoformat(), live=live, source_checks=identities,
                     current_native_progress_evidence=ref(runtime_path),
                     command='python -B scripts/finalize_y58_y59_handoff.py --token ' + args.token,
                     explanation='Authoritative completed fixed30 checkpoint/log and absent exact old '
                                 'process, not fabricated deathwatch counts; preserve already running Y59.')
        tick.save(base.with_suffix('.json'), proof)
        destination = ROOT / ('Checkpoints/server-2/' + basename + '/' + args.token + '/fixed30.log')
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        subprocess.run(['scp', '-q', server['user'] + '@' + server['host'] + ':' + old['log'],
                        str(destination)], check=True, timeout=60)
        assert ref(destination)['sha256'] == proof['log_sha256']
        lease(args.token, 120)
        original_pull = monitor.pull_log
        def verified_pull(project_dir, host, user, remote_log, cfg_name, state, srv_name=''):
            assert Path(project_dir) == ROOT and host == server['host'] and user == server['user']
            assert remote_log == old['log'] and cfg_name == basename
            assert state == 'done_complete' and srv_name == 'server-2'
            return str(destination)
        monitor.pull_log = verified_pull
        events = []
        try:
            monitor.finalize_task(str(ROOT), queue, 'server-2', server['host'], server['user'], old,
                basename, 'done_complete', proof['checkpoints'][0]['best_miou'],
                '30/30', '0', old['log'], events)
        finally:
            monitor.pull_log = original_pull
        experiment = monitor.load_experiment(str(ROOT), old['id'])
        experiment.update(terminal_evidence=ref(base.with_suffix('.json')),
                          terminal_checkpoint_metadata=proof['checkpoints'])
        monitor.save_experiment(str(ROOT), experiment)
        deathwatch = monitor.load_deathwatch(str(ROOT))
        deathwatch.get('server-2', {}).pop(basename, None)
        monitor.save_deathwatch(str(ROOT), deathwatch)
        additions = [ref(base.with_suffix('.json')), ref(base.with_suffix('.log')),
                     ref(runtime_path), ref(__file__)]
        for candidate in candidates.values():
            candidate['evidence'].extend(entry for entry in additions if entry not in candidate['evidence'])
        candidates[old['id']].update(stage='blocked', blocked_from_stage='validation',
            scientific_status='fixed30_complete_frozen_review_pending', required_server='server-2',
            blocker_kind='exclusive_gpu_slot', occupation_evidence=ref(runtime_path),
            blocker='Y58 fixed30 training is complete; the sole authorized GPU0 now runs admitted Y59. '
                    'Frozen full evaluation is pending, not a reason to restart or promote Y58.',
            retry_after=(observed_at + timedelta(hours=1)).isoformat(),
            wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
            unblock_condition='Canonical queue/running opens GPU0 and fresh live checks confirm exclusive idle use.',
            next_action='Preserve completed fixed30 checkpoints and original lineage. Execute the registered '
                'full frozen evaluation/comparison and trained costs on exclusive authorized GPU0 without '
                'tuning or restarting. A training log best is not a scientific promotion.')
        candidates[current['id']].update(stage='training',
            next_action='Monitor this exact verified fixed30 Y59 native run and preserve checkpoint/log '
                        'lineage. Keep seed1999 and the admitted source; evaluate efficacy only through '
                        'the frozen comparison, without validation tuning.')
        for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after',
                    'unblock_condition', 'occupation_evidence'):
            candidates[current['id']].pop(key, None)
        work['tick_attempt'] = dict(
            tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id=current['id'], kind='resource_check', result='passed',
            command=proof['command'], observation='Y58 exact last checkpoint epoch30/model/optimizer/'
                'scheduler and terminal log verified; canonical archive completed. Actual Y59 GPU0 native '
                'steps grow and its original receipt/source matches. No new smoke, admission or efficacy.',
            evidence=[ref(base.with_suffix('.json')), ref(runtime_path)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        after = monitor.load_queue(str(ROOT))['gpu_queues']['server-2']
        assert not any(t['id'] == old['id'] for t in after)
        assert next(t for t in after if t['id'] == current['id']) == current
        running = monitor.load_running(str(ROOT))['server-2']['0']
        assert current['id'] == running or isinstance(running, list) and running == [current['id']]
        print(json.dumps(dict(proof=ref(base.with_suffix('.json')), canonical_state=experiment['status'],
            checkpoints=proof['checkpoints'], current_run=current['id'], current_pid=current['pid'],
            Y57_unchanged_paused=True, events=events), indent=2))
    finally:
        writer.close()


if __name__ == '__main__':
    main()
