"""Read back the approved limit and original training progress without changing hardware."""

import argparse
import csv
from datetime import datetime
import io
import json
import re
import shlex

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import closure
from dispatch_prepared_candidates import snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 120)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        applied_path = ROOT / ('pipeline/checks/gpu0_runtime_power_authenticated_' + args.token + '.json')
        applied = tick.load(applied_path)
        assert applied['complete'] and applied['gpu0_limit_verified'] and applied['gpu1_limit_unchanged']
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0, 1]
        queue = monitor.load_queue(str(ROOT))
        tasks = [t for t in queue['gpu_queues']['server-2'] if t['status'] in ('running', 'claiming')]
        live = snapshot(server, tasks, [])
        assert live['gpu_query_rc'] == live['compute_query_rc'] == 0
        health_program = """
import json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
out={'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
for name,cmd in {
 'devices':['nvidia-smi','--query-gpu=index,uuid,power.limit,power.draw,temperature.gpu,fan.speed,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
 'performance':['nvidia-smi','-q','-d','PERFORMANCE'],
 'kernel':['dmesg','-T']}.items():
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
 text=r.stdout
 if name=='kernel':text='\\n'.join(v for v in text.splitlines() if 'NVRM: Xid' in v or 'fallen off' in v)
 out[name]={'rc':r.returncode,'stdout':text,'stderr':r.stderr}
print(json.dumps(out))
"""
        health = json.loads(monitor.ssh(server['host'], server['user'],
            shlex.quote(server['python']) + ' -B -c ' + shlex.quote(health_program), timeout=45))
        assert health['devices']['rc'] == 0
        devices = {int(row[0]): dict(uuid=row[1].strip(), limit_watts=float(row[2]),
            draw_watts=float(row[3]), temperature_c=float(row[4]), fan_percent=float(row[5]),
            utilization_percent=float(row[6]), memory_used_mib=int(row[7]), memory_total_mib=int(row[8]))
            for row in csv.reader(io.StringIO(health['devices']['stdout']))}
        assert set(devices) == {0, 1}
        expected = applied['after']
        hardware_verified = (health['boot_id'] == applied['boot_id']
            and devices[0]['uuid'] == expected['0']['uuid'] and devices[0]['limit_watts'] == 280
            and devices[1]['uuid'] == expected['1']['uuid'] and devices[1]['limit_watts'] == 390)
        progress = []
        for task in tasks:
            row = live['tasks'][task['id']]
            steps = closure.steps(row['tail'])
            real = (row['process_live'] and row['cwd'] == server['remote_code_root']
                    and task['config'] in row['cmdline'] and 'tools/train.py' in row['cmdline'])
            epochs = re.findall(r'(Train|Val)\[(\d+)/(\d+)\]', row['tail'])
            progress.append(dict(id=task['id'], gpu=task.get('gpu'), pid=task.get('pid'),
                process_verified=bool(real), native_progress_verified=bool(real and len(set(steps)) >= 2),
                phase=epochs[-1][0] if epochs else None, epoch=int(epochs[-1][1]) if epochs else None,
                total_epochs=int(epochs[-1][2]) if epochs else None,
                first_step=steps[0] if steps else None, last_step=steps[-1] if steps else None,
                observed_error=any(s in row['tail'] for s in ('Traceback', 'CUDA out of memory', 'RuntimeError'))))
        no_new_kernel_error = health['kernel']['rc'] == 0 and not health['kernel']['stdout']
        complete = hardware_verified and no_new_kernel_error and len(progress) == 2 and all(
            item['native_progress_verified'] and not item['observed_error'] for item in progress)
        output = ROOT / ('pipeline/checks/gpu0_post_power_verify_' + args.token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=complete, observed_at=datetime.now().astimezone().isoformat(),
            applied=ref(applied_path), live=live, health=health, devices=devices, training=progress,
            hardware_verified=hardware_verified, no_new_kernel_error=no_new_kernel_error,
            mutation_performed=False, limitation='A short thermal/progress observation is not proof '
            'of steady-state cooling, permanent repair, a resolved hardware cause or scientific efficacy.'))
        if complete:
            work = tick.load(ROOT / 'pipeline/research_work.json')
            candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
            candidate['evidence'].append(ref(output))
            candidate.setdefault('runtime_hardware_verifications', []).append(ref(output))
            tick.save(ROOT / 'pipeline/research_work.json', work)
            experiment_path = ROOT / 'pipeline/experiments/AdaRES-Y56.json'
            experiment = tick.load(experiment_path)
            experiment.setdefault('runtime_hardware_verifications', []).append(ref(output))
            tick.save(experiment_path, experiment)
        print(json.dumps(dict(complete=complete, evidence=ref(output), devices=devices,
            training=progress, no_new_kernel_error=no_new_kernel_error), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
