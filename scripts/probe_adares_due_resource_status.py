"""Read current authorized resource state and independently sample Y50 steps."""
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import time

assert P['physical_gpu'] is None
root = Path(P['root'])
log_path = root / 'logs/train_adares_y50_joint_incidence_res_20261016_145628.log'


def log_tail(limit):
    with log_path.open('rb') as stream:
        stream.seek(0, 2)
        length = stream.tell()
        stream.seek(max(0, length - limit))
        return stream.read(limit).decode('utf-8', errors='replace')


def progress():
    text = log_tail(32768)
    values = re.findall(r'Train\[(\d+)/(\d+)\]:[^\r\n]*?(\d+)/(\d+)\s*\[', text)
    return list(map(int, values[-1])) if values else None


before = progress()
gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits']).decode()
process_text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits']).decode()
compute = [dict(pid=int(row[0].strip()), gpu_uuid=row[1].strip(), memory_mib=int(row[2].strip()))
           for row in csv.reader(io.StringIO(process_text)) if row]
gpus = []
for row in csv.reader(io.StringIO(gpu_text)):
    if not row or int(row[0].strip()) not in (0, 1):
        continue
    index, uuid, utilization, memory = int(row[0]), row[1].strip(), int(row[2]), int(row[3])
    pids = [item['pid'] for item in compute if item['gpu_uuid'] == uuid]
    gpus.append(dict(gpu=index, uuid=uuid, utilization_percent=utilization, memory_mib=memory,
                     compute_pids=pids, eligible_for_exclusive_diagnostic=(not pids and memory < 1024 and utilization < 10)))
assert {row['gpu'] for row in gpus} == {0, 1}
pid = subprocess.run(['ps', '-ww', '-p', '31539', '-o', 'pid=,args='], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
command = pid.stdout.decode().strip()
expected_command = 'tools/train.py configs_song/model_paper5/adares_y50_joint_incidence_res.yaml'
same_process = pid.returncode == 0 and expected_command in command and '--num_gpus 1 --gpu_ids 1' in command
time.sleep(2.)
after = progress()
advanced = (after[0], after[2]) > (before[0], before[2]) if before and after else None
text = log_tail(8388608)
metric_rows = [dict(miou_rounded=float(a), acc50_rounded=float(b), acc25_rounded=float(c))
               for a, b, c in re.findall(r'(?<![A-Za-z])mIOU:\s*([0-9.]+)\.\s*Acc_50:\s*([0-9.]+)\.\s*Acc_25:\s*([0-9.]+)', text)]
sp_rows = [dict(spmiou_rounded=float(a), spacc50_rounded=float(b), spacc25_rounded=float(c))
           for a, b, c in re.findall(r'spmIOU:\s*([0-9.]+)\.\s*spAcc_50:\s*([0-9.]+)\.\s*spAcc_25:\s*([0-9.]+)', text)]
result = dict(complete=True, project='AdaRES_project', no_GPU=True, read_only=True,
    sampling='Two independent bounded log reads separated by two seconds; use client execution window, not server clock.',
    resources=gpus, idle_eligible_gpu_ids=[row['gpu'] for row in gpus if row['eligible_for_exclusive_diagnostic']],
    GPU_lease_acquired=False, no_resource_reservation=True,
    Y50=dict(pid=31539, same_process_confirmed=same_process, command=command if same_process else None,
             process_exit_code=pid.returncode, progress_before=before, progress_after=after,
             native_steps_advanced=advanced, latest_rounded_validation=metric_rows[-1] if metric_rows else None,
             latest_rounded_superpoint_validation=sp_rows[-1] if sp_rows else None,
             crash_marker_in_last_32768_bytes=any(value in text[-32768:] for value in ('Traceback (most recent call last)', 'CUDA out of memory'))),
    limits=['Eligibility is a live observation, not a lease or admission; recheck and lock before GPU use.',
            'PID existence without increasing native steps is not proof of current training progress.',
            'Rounded early validation values are not a completed trained comparison or efficacy result.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
