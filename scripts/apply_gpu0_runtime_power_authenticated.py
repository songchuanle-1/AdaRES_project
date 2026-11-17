"""Retry a confirmed unchanged power setting using memory-only sudo authentication."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick


def private_password():
    reference = Path('C:/Users/chuanle song/.codex/private/server-accounts.md')
    matches = []
    for line in reference.read_text(encoding='utf-8').splitlines():
        if '|' not in line:
            continue
        cells = next(csv.reader([line], delimiter='|'))
        if len(cells) != 8:
            continue
        cells = [cell.strip().strip('`') for cell in cells[1:-1]]
        if cells[1] != '192.168.1.163':
            continue
        for account_index in (2, 4):
            if cells[account_index] == 'songcl':
                matches.append(cells[account_index + 1])
    assert len(matches) == 1 and matches[0] and '\n' not in matches[0] and '\r' not in matches[0], \
        'No unambiguous private credential for this account; do not print private source'
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    auth_path = ROOT / ('pipeline/authorizations/GPU0_runtime_power_280W_' + args.token + '.json')
    authorization = tick.load(auth_path)
    assert authorization['authorization_granted'] and authorization['human_reply'] == '允许仅 GPU0 临时降至280W'
    assert authorization['host'] == '192.168.1.163' and authorization['gpu_index'] == 0
    assert authorization['gpu_uuid'] == 'GPU-e0b11432-b7aa-2006-1903-dcc739f26146'
    assert authorization['temporary_limit_watts'] == 280 and authorization['runtime_only']
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == authorization['host'] and server['allowed_gpus'] == [0, 1]
        queue = monitor.load_queue(str(ROOT))
        task = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y56')
        assert task['status'] == 'running' and str(task['gpu']) == '0' and task['pid'] == 13738
        prior_path = ROOT / ('pipeline/checks/gpu0_runtime_power_' + args.token + '.json')
        prior = tick.load(prior_path)
        assert prior['complete'] is False and prior['change_returncode'] == 1
        assert 'password is required' in prior['change_stderr']
        assert prior['after']['0']['limit_watts'] == 390 and prior['after']['1']['limit_watts'] == 390
        output = ROOT / ('pipeline/checks/gpu0_runtime_power_authenticated_' + args.token + '.json')
        assert not output.exists(), 'An uncertain prior change requires read-only reconciliation, not blind retry'
        parameters = dict(uuid=authorization['gpu_uuid'], old_limit=390.0, new_limit=280.0,
            boot_id='81549f32-5f49-4ef5-b91e-2b1144a6bfb3', pid=task['pid'],
            config=task['config'], code_root=server['remote_code_root'])
        program = 'P=' + repr(parameters) + '\n' + r'''
import csv,io,json,pathlib,shutil,subprocess,datetime,sys
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot==P['boot_id'],'Boot changed; reconcile before modifying hardware'
nv=shutil.which('nvidia-smi')
assert nv and pathlib.Path(nv).is_absolute()
def devices():
 r=subprocess.run([nv,'--query-gpu=index,uuid,power.limit,power.min_limit,power.max_limit,temperature.gpu',
  '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 assert r.returncode==0,r.stderr
 rows={int(row[0]):row for row in csv.reader(io.StringIO(r.stdout))}
 assert set(rows)=={0,1}
 return {i:dict(index=i,uuid=row[1].strip(),limit_watts=float(row[2]),min_watts=float(row[3]),
  max_watts=float(row[4]),temperature_c=float(row[5])) for i,row in rows.items()}
before=devices()
assert before[0]['uuid']==P['uuid'] and before[0]['limit_watts']==P['old_limit']
assert before[0]['min_watts']<=P['new_limit']<=before[0]['max_watts']
proc=pathlib.Path('/proc')/str(P['pid'])
argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
assert P['config'] in argv and 'tools/train.py' in argv
assert str((proc/'cwd').resolve())==P['code_root']
compute=subprocess.run([nv,'-i',P['uuid'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],
 capture_output=True,text=True,timeout=10)
assert compute.returncode==0 and compute.stdout.strip().splitlines()==[str(P['pid'])], 'GPU owner changed'
command=['sudo','-S','-p','','--',nv,'-i',P['uuid'],'-pl',str(int(P['new_limit']))]
credential=sys.stdin.buffer.readline()
assert credential and credential.endswith(b'\n')
change=subprocess.run(command,input=credential,capture_output=True,timeout=15)
del credential
after=devices()
verified=(change.returncode==0 and after[0]['uuid']==P['uuid'] and
 after[0]['limit_watts']==P['new_limit'] and after[1]['uuid']==before[1]['uuid'] and
 after[1]['limit_watts']==before[1]['limit_watts'] and
 pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot)
print(json.dumps(dict(complete=verified,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 hostname='dl01-MS-7C60',boot_id=boot,before=before,after=after,command=command,
 change_returncode=change.returncode,change_stdout=change.stdout.decode(errors='replace'),change_stderr=change.stderr.decode(errors='replace'),
 gpu0_limit_verified=after[0]['limit_watts']==P['new_limit'],
 gpu1_limit_unchanged=after[1]['limit_watts']==before[1]['limit_watts'],
 runtime_only=True,training_restarted=False,persistent_config_changed=False)))
'''
        secret = private_password()
        response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
            server['user'] + '@' + server['host'],
            shlex.quote(server['python']) + ' -B -c ' + shlex.quote(program)],
            input=secret + '\n', capture_output=True, text=True, timeout=60)
        stdout = response.stdout.replace(secret, '[redacted]')
        stderr = response.stderr.replace(secret, '[redacted]')
        del secret
        assert response.returncode == 0, 'Authenticated command failed before verification: ' + stderr
        result = json.loads(stdout)
        result.update(authorization=ref(auth_path), helper=ref(__file__),
                      recorded_at=datetime.now().astimezone().isoformat(),
                      prior_confirmed_no_change=ref(prior_path), credential_saved_or_logged=False,
                      key_based_SSH_preserved=True)
        tick.save(output, result)
        if not result['complete']:
            print(json.dumps(dict(verified=False, evidence=ref(output), actual=result), indent=2))
            raise RuntimeError('Power change not fully verified; preserve evidence and do not blindly retry')
        event = dict(observed_at=result['observed_at_utc'], server='server-2', gpu=0,
            gpu_uuid=authorization['gpu_uuid'], previous_limit_watts=390, runtime_limit_watts=280,
            authorization=ref(auth_path), evidence=ref(output),
            note='Explicit human approval; only runtime GPU0 power limit changed. '
                 'GPU1 power, model/data/optimizer/seed/fixed30 and original process unchanged. '
                 'Future timing/resource comparisons must retain this hardware-condition change.')
        experiment_path = ROOT / 'pipeline/experiments/AdaRES-Y56.json'
        experiment = tick.load(experiment_path)
        experiment.setdefault('runtime_hardware_history', []).append(event)
        tick.save(experiment_path, experiment)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
        candidate.setdefault('runtime_hardware_history', []).append(event)
        for entry in (ref(auth_path), ref(output)):
            if entry not in candidate['evidence']:
                candidate['evidence'].append(entry)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(verified=True, evidence=ref(output), before=result['before'],
            after=result['after'], gpu1_limit_unchanged=result['gpu1_limit_unchanged']), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
