"""Observe the protected single-card trial without changing remote state."""

import argparse
import csv
from datetime import datetime
import io
import json
import shlex

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick


REMOTE = r'''
import csv, io, json, pathlib, re, subprocess, time

assert subprocess.check_output(['hostname']).decode().strip() == 'dl01-MS-7C60'
boot = pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
expected_uuid = 'GPU-e0b11432-b7aa-2006-1903-dcc739f26146'

def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    return dict(rc=result.returncode, stdout=result.stdout, stderr=result.stderr)

def sample():
    proc = pathlib.Path('/proc') / str(TASK['pid'])
    live = (proc / 'cmdline').is_file()
    argv = [v.decode(errors='replace') for v in
            (proc / 'cmdline').read_bytes().split(b'\0') if v] if live else []
    cwd = str((proc / 'cwd').resolve()) if live else None
    starttime = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19] if live else None
    log = pathlib.Path(TASK['log'])
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 24000))
        tail = stream.read().decode('utf-8', errors='replace')
    matches = re.findall(r'(Train|Val)\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)', tail)
    progress = matches[-1] if matches else None
    telemetry = run(['nvidia-smi', '-i', expected_uuid,
        '--query-gpu=index,uuid,power.limit,power.draw,temperature.gpu,fan.speed,utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits'])
    rows = list(csv.reader(io.StringIO(telemetry['stdout']))) if telemetry['rc'] == 0 else []
    device = None
    if len(rows) == 1:
        row = [v.strip() for v in rows[0]]
        device = dict(index=int(row[0]), uuid=row[1], power_limit_watts=float(row[2]),
            draw_watts=float(row[3]), temperature_c=float(row[4]), fan_percent=float(row[5]),
            utilization_percent=float(row[6]), memory_used_mib=int(row[7]), memory_total_mib=int(row[8]))
    return dict(process_live=live, cmdline=argv, cwd=cwd, starttime=starttime,
        log_size_bytes=log.stat().st_size, progress=progress, device=device,
        telemetry=telemetry, monotonic=time.monotonic(),
        tail_errors=[v for v in ('Traceback', 'RuntimeError', 'CUDA out of memory') if v in tail])

first = sample()
time.sleep(12)
second = sample()
performance = run(['nvidia-smi', '-i', expected_uuid, '-q', '-d', 'PERFORMANCE'])
compute = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
    '--format=csv,noheader,nounits'])
kernel = run(['dmesg', '-T'])
kernel['stdout'] = '\n'.join(v for v in kernel['stdout'].splitlines()
    if 'NVRM: Xid' in v or 'fallen off' in v)
print(json.dumps(dict(hostname='dl01-MS-7C60', boot_id=boot, samples=[first, second],
    performance=performance, compute=compute, kernel=kernel)))
'''


def evaluate(observation, task, root):
    samples = observation['samples']
    identity = all(s['process_live'] and s['cwd'] == root and
        s['cmdline'].count(task['config']) == 1 and 'tools/train.py' in s['cmdline']
        for s in samples)
    same_process = identity and samples[0]['starttime'] == samples[1]['starttime']
    progress = same_process and all(s['progress'] for s in samples) and (
        samples[0]['progress'] != samples[1]['progress'] and
        samples[1]['log_size_bytes'] > samples[0]['log_size_bytes'])
    cap = all(s['device'] and s['device']['uuid'] == 'GPU-e0b11432-b7aa-2006-1903-dcc739f26146'
              and s['device']['index'] == 0 and s['device']['power_limit_watts'] == 280
              for s in samples)
    compute = observation['compute']
    rows = list(csv.reader(io.StringIO(compute['stdout'])))
    gpu1_empty = compute['rc'] == 0 and not any(
        row[0].strip() == 'GPU-f740248b-4c9e-c8bb-b14b-99d0b9eb6c79' for row in rows)
    gpu0_owned = compute['rc'] == 0 and any(
        row[0].strip() == 'GPU-e0b11432-b7aa-2006-1903-dcc739f26146'
        and int(row[1]) == int(task['pid']) for row in rows)
    performance = observation['performance']
    thermal = None
    if performance['rc'] == 0:
        rows = [v.strip() for v in performance['stdout'].splitlines() if 'Thermal Slowdown' in v]
        thermal = any(v.split(':', 1)[1].strip() == 'Active' for v in rows) if rows else None
    kernel = observation['kernel']
    kernel_errors = bool(kernel['stdout']) if kernel['rc'] == 0 else None
    no_tail_error = not any(s['tail_errors'] for s in samples)
    healthy = bool(progress and cap and gpu1_empty and gpu0_owned and no_tail_error
                   and kernel_errors is False and thermal is False)
    return dict(process_identity_verified=bool(same_process), native_progress_verified=bool(progress),
        power_cap_preserved=cap, gpu1_compute_empty=gpu1_empty, gpu0_compute_owner_verified=gpu0_owned,
        thermal_slowdown_active=thermal, kernel_xid_or_fallen_off_observed=kernel_errors,
        tail_error_observed=not no_tail_error, sample_healthy=healthy,
        hardware_root_cause_resolved=False, efficacy_test=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 150)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None, 'Another project writer is active'
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        queue = monitor.load_queue(str(ROOT))
        tasks = queue['gpu_queues']['server-2']
        task = next(t for t in tasks if t['id'] == 'AdaRES-Y56')
        paused = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
        assert task['status'] == 'running' and int(task['gpu']) == 0 and int(task['pid']) == 13738
        assert paused['status'] == 'paused' and paused['dispatch_hold']
        assert paused['resume_requires_user_authorization']
        output = ROOT / ('pipeline/checks/gpu0_single_card_' + args.token + '.json')
        assert not output.exists(), 'Do not replace an existing observation'
        program = 'TASK=' + repr(dict(pid=task['pid'], config=task['config'], log=task['log'])) + '\n' + REMOTE
        observed = json.loads(monitor.ssh(server['host'], server['user'],
            shlex.quote(server['python']) + ' -B -c ' + shlex.quote(program), timeout=75))
        assessment = evaluate(observed, task, server['remote_code_root'])
        result = dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
            candidate_id=task['id'], command='python -B scripts/check_gpu0_single_card_trial.py --token ' + args.token,
            observation=observed, assessment=assessment, y57_registered_paused=True,
            remote_mutations=False, hardware_settings_changed=False, queue_changed=False,
            limitation='Two samples only. Not a sustained stress test, causal cooling attribution, '
                       'native smoke for prepared candidates, hardware repair or trained efficacy result.')
        lease(args.token, 60)
        tick.save(output, result)
        evidence = ref(output)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == task['id'])
        for entry in (evidence, ref(__file__)):
            if entry not in candidate['evidence']:
                candidate['evidence'].append(entry)
        work['tick_attempt'] = dict(
            tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id=task['id'], kind='resource_check',
            result='passed' if assessment['sample_healthy'] else 'blocked', command=result['command'],
            observation='Observed protected Y56 GPU0-only training with two log/telemetry samples; '
                + json.dumps(assessment, sort_keys=True) + '. No prepared-candidate retry or GPU workload launched.',
            evidence=[evidence])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(evidence=evidence, observed_at=result['observed_at'], assessment=assessment,
            samples=[dict(progress=s['progress'], device=s['device']) for s in observed['samples']]), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
