"""Pause only the verified Y57 process and preserve the continuing Y56 run."""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import shlex
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

REMOTE = r'''
import csv,gc,hashlib,io,json,os,pathlib,re,signal,subprocess,time
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='81549f32-5f49-4ef5-b91e-2b1144a6bfb3'
root=pathlib.Path(P['root']).resolve()
work=pathlib.Path(P['work']).resolve()
def digest(path):
 h=hashlib.sha256()
 with open(str(path),'rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def process(pid):
 path=pathlib.Path('/proc')/str(pid)
 try:
  stat=(path/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,state=stat[0],ppid=int(stat[1]),start=stat[19],
   argv=[v.decode(errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v],
   cwd=str((path/'cwd').resolve()),uid=path.stat().st_uid)
 except (OSError,ValueError):return None
def tail(path):
 with path.open('rb') as stream:
  stream.seek(max(0,path.stat().st_size-24000))
  return stream.read().decode('utf-8',errors='replace')
def devices():
 r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total',
  '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 assert r.returncode==0,r.stderr
 return {int(v[0]):v for v in csv.reader(io.StringIO(r.stdout))}
def apps(gpu):
 r=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],
  capture_output=True,text=True,timeout=10)
 assert r.returncode==0,r.stderr
 return [int(v.strip()) for v in r.stdout.splitlines() if v.strip()]
target=process(P['pid'])
keeper=process(P['keep_pid'])
assert target and keeper and target['uid']==keeper['uid']==os.getuid()
for row,config,gpu in ((target,P['config'],'1'),(keeper,P['keep_config'],'0')):
 assert row['cwd']==str(root) and 'tools/train.py' in row['argv'] and config in row['argv']
 assert row['argv'][row['argv'].index('--gpu_ids')+1]==gpu
 assert row['argv'][row['argv'].index('--num_gpus')+1]=='1'
assert apps(1)==[P['pid']]
before=devices()
assert before[0][1].strip()=='GPU-e0b11432-b7aa-2006-1903-dcc739f26146'
assert before[1][1].strip()=='GPU-f740248b-4c9e-c8bb-b14b-99d0b9eb6c79'
log=pathlib.Path(P['log']).resolve()
log.relative_to(root/'logs')
with log.open('rb') as stream:
 stream.seek(max(0,log.stat().st_size-4*1024*1024))
 text=stream.read().decode('utf-8',errors='replace')
paths=re.findall(r'save last pth to ([^\s]+/last\.pth)',text)
assert paths,'No completed last-checkpoint announcement in the original run log'
checkpoint=pathlib.Path(paths[-1]).resolve()
checkpoint.relative_to(work/pathlib.Path(P['config']).stem)
best=checkpoint.with_name('best.pth')
assert checkpoint.is_file() and best.is_file() and checkpoint.stat().st_size>100000000
stat_before=(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
import torch
torch.set_num_threads(2)
c=torch.load(str(checkpoint),map_location='cpu')
assert all(k in c for k in ('model','optimizer','scheduler','meta'))
epoch=int(c['meta']['epoch'])
assert 0<epoch<30,'Run is already complete; reconcile rather than pause it'
assert c['optimizer']['state'] and c['optimizer']['param_groups']
assert int(c['scheduler']['last_epoch'])==epoch
meta=dict(epoch=epoch,best_miou=float(c['meta']['best_miou']),
 optimizer_state_count=len(c['optimizer']['state']),scheduler_last_epoch=int(c['scheduler']['last_epoch']))
del c
gc.collect()
checkpoint_sha=digest(checkpoint)
best_sha=digest(best)
assert stat_before==(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns),'Checkpoint writer changed; retry inspection'
fresh=process(P['pid'])
assert fresh and fresh['start']==target['start'] and fresh['argv']==target['argv']
all_processes={}
for entry in pathlib.Path('/proc').iterdir():
 if entry.name.isdigit():
  row=process(int(entry.name))
  if row:all_processes[row['pid']]=row
descendants={}
parents={P['pid']}
while True:
 new={pid:row for pid,row in all_processes.items() if row['ppid'] in parents and pid not in parents}
 if not new:break
 descendants.update(new)
 parents.update(new)
journal=pathlib.Path(P['journal'])
journal.parent.mkdir(parents=True,exist_ok=True)
with journal.open('x') as stream:
 json.dump(dict(token=P['token'],phase='verified_before_SIGTERM',target=target,keeper=keeper,
  checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,meta=meta),stream)
os.kill(P['pid'],signal.SIGTERM)
for _ in range(160):
 row=process(P['pid'])
 if not row or row['state']=='Z':break
 time.sleep(.1)
row=process(P['pid'])
assert not row or row['state']=='Z','Target still live after SIGTERM; no broad kill or automatic restart'
for pid,old in descendants.items():
 row=process(pid)
 if row and row['state']!='Z' and row['start']==old['start'] and row['uid']==os.getuid():
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
for _ in range(100):
 remaining=[]
 for pid,old in descendants.items():
  row=process(pid)
  if row and row['state']!='Z' and row['start']==old['start']:remaining.append(pid)
 if not remaining:break
 time.sleep(.1)
assert not remaining,'Some verified children remain; preserve intent and reconcile'
for _ in range(60):
 remaining_apps=apps(1)
 if not remaining_apps:break
 time.sleep(.2)
assert not remaining_apps,'GPU1 compute still occupied; do not claim released'
keep_after=process(P['keep_pid'])
assert keep_after and keep_after['start']==keeper['start'] and keep_after['argv']==keeper['argv']
assert keep_after['cwd']==keeper['cwd']
assert digest(checkpoint)==checkpoint_sha and digest(best)==best_sha,'Saved checkpoint changed around stop'
after=devices()
assert after[0][1:3]==before[0][1:3] and after[1][1:3]==before[1][1:3]
assert float(after[1][5])<1024,'GPU1 memory not released'
progress=re.findall(r'(Train|Val)\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)',tail(log))
print(json.dumps(dict(complete=True,stopped_pid=P['pid'],signal='SIGTERM',descendant_pids=sorted(descendants),
 keeper_pid=P['keep_pid'],keeper_process_unchanged=True,gpu1_compute_pids=remaining_apps,
 checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,best_checkpoint=str(best),best_sha256=best_sha,
 checkpoint_meta=meta,resume_epoch=epoch+1,original_log=str(log),original_log_sha256=digest(log),
 last_progress=progress[-1] if progress else None,before_devices=before,after_devices=after,
 preserved_model_optimizer_scheduler=True,training_protocol_changed=False,hardware_settings_changed=False,
 note='Latest fully saved checkpoint preserved. Unsaved current-epoch progress/RNG state is not promised recovered.')))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 300)
    auth_path = ROOT / ('pipeline/authorizations/GPU0_only_trial_' + args.token + '.json')
    auth = tick.load(auth_path)
    assert auth['authorization_granted'] and auth['human_reply'] == 'y57断开吧  先试用单卡GPU0跑'
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0, 1]
        queue = monitor.load_queue(str(ROOT))
        tasks = queue['gpu_queues']['server-2']
        target = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
        keeper = next(t for t in tasks if t['id'] == 'AdaRES-Y56')
        assert target['status'] == keeper['status'] == 'running'
        assert target['pid'] == 16317 and keeper['pid'] == 13738
        assert str(target['gpu']) == '1' and str(keeper['gpu']) == '0'
        base = ROOT / ('pipeline/checks/y57_user_pause_' + args.token)
        assert not base.with_suffix('.json').exists() and not base.with_suffix('.log').exists()
        target.update(dispatch_hold='User pause requested: GPU0-only trial; no automatic Y57 resume',
            user_pause_intent=dict(authorization=ref(auth_path),token=args.token,
                pid=target['pid'],config=target['config'],request='pause_pending_verification'))
        monitor.save_queue(str(ROOT), queue)
        payload = dict(token=args.token, pid=target['pid'], config=target['config'], log=target['log'],
            keep_pid=keeper['pid'], keep_config=keeper['config'], root=server['remote_code_root'],
            work=server['work_dir'], journal='/bak_home/songcl/Project/AdaRES_project/pipeline/probes/' +
                args.token + '/y57_pause_before_signal.json')
        program = 'P=' + repr(payload) + '\n' + REMOTE
        result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
            server['user'] + '@' + server['host'],
            'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=' + shlex.quote(server['site_packages']) + ' ' +
            shlex.quote(server['python']) + ' -S -B -'],
            input=program.encode('utf-8'),capture_output=True,timeout=180)
        base.with_suffix('.log').write_bytes(result.stdout + b'\nSTDERR:\n' + result.stderr)
        assert result.returncode == 0, 'Pause command incomplete; preserve raw log and durable intent, do not blindly retry'
        evidence = json.loads(result.stdout)
        assert evidence['complete'] and evidence['keeper_process_unchanged']
        evidence.update(observed_at=datetime.now().astimezone().isoformat(),authorization=ref(auth_path),helper=ref(__file__))
        log_copy = ROOT / 'Checkpoints/server-2/adares_y57_peer_second_moment_res' / args.token / 'user_paused.log'
        log_copy.parent.mkdir(parents=True,exist_ok=True)
        assert not log_copy.exists()
        subprocess.run(['scp','-q',server['user'] + '@' + server['host'] + ':' + evidence['original_log'],
            str(log_copy)],check=True,timeout=45)
        assert ref(log_copy)['sha256'] == evidence['original_log_sha256']
        evidence['local_log_copy'] = ref(log_copy)
        tick.save(base.with_suffix('.json'),evidence)
        proof = ref(base.with_suffix('.json'))
        target.update(status='paused',paused_at=evidence['observed_at'],pause_evidence=proof,
            resume_checkpoint=evidence['checkpoint'],resume_checkpoint_sha256=evidence['checkpoint_sha256'],
            resume_requires_user_authorization=True,resume_epoch=evidence['resume_epoch'])
        target['user_pause_intent']['request']='pause_verified'
        monitor.save_queue(str(ROOT),queue)
        running=monitor.load_running(str(ROOT))
        assert running['server-2'].get('1')=='AdaRES-Y57' and running['server-2'].get('0')=='AdaRES-Y56'
        running['server-2'].pop('1')
        monitor.save_running(str(ROOT),running)
        experiment=monitor.load_experiment(str(ROOT),'AdaRES-Y57')
        experiment.update(status='paused',paused_at=evidence['observed_at'],pause_evidence=proof,
            resume_checkpoint=evidence['checkpoint'],resume_requires_user_authorization=True,
            dispatch_hold=target['dispatch_hold'])
        experiment.setdefault('history',[]).append(dict(ts=evidence['observed_at'],action='user_paused',
            gpu='1',pid=16317,checkpoint=evidence['checkpoint'],saved_epoch=evidence['checkpoint_meta']['epoch'],
            evidence=proof,authorization=ref(auth_path),scientifically_complete=False))
        monitor.save_experiment(str(ROOT),experiment)
        work=tick.load(ROOT/'pipeline/research_work.json')
        candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y57')
        candidate.update(stage='blocked',blocked_from_stage='training',blocker_kind='human_authorization',
            blocker='The user explicitly paused Y57/GPU1 for a GPU0-only trial. Not a rejection, crash or completed fixed30 run.',
            retry_after=(datetime.now().astimezone()+timedelta(hours=4)).isoformat(),
            unblock_condition='Explicit user instruction to resume this SAME experiment on an authorized free resource; no automatic resume or GPU1 use.',
            next_action='Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. '
                'Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; '
                'keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0.',
            user_pause_authorization=ref(auth_path),pause_evidence=proof)
        candidate['evidence'].extend([ref(auth_path),proof,ref(base.with_suffix('.log'))])
        work['tick_attempt']=dict(
            tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id='AdaRES-Y57',kind='resource_check',result='blocked',
            command='python -B scripts/pause_y57_gpu0_trial.py --token '+args.token,
            observation='User-authorized Y57 SIGTERM and GPU1 release verified; latest saved checkpoint '
                'model/optimizer/scheduler loaded and hashed, original Y56 PID preserved. '
                'Y57 awaits explicit user resume instruction, not new native admission.',evidence=[proof])
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(evidence=proof,paused_experiment='AdaRES-Y57',stopped_pid=16317,
            keeper_pid=13738,checkpoint_meta=evidence['checkpoint_meta'],checkpoint=evidence['checkpoint'],
            gpu1_compute_pids=evidence['gpu1_compute_pids'],devices=evidence['after_devices']),indent=2))
    finally:
        lock.close()


if __name__=='__main__':
    main()
