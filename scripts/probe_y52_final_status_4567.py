"""Read-only live occupation and bounded existing-training evidence."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

assert P['physical_gpu'] is None
root = Path(P['root'])
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits']).decode().splitlines()
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits']).decode().splitlines()
pid = subprocess.check_output(['ps', '-ww', '-p', '31539', '-o', 'pid=,args=']).decode().strip()
assert 'tools/train.py configs_song/model_paper5/adares_y50_joint_incidence_res.yaml' in pid
assert '--num_gpus 1 --gpu_ids 1' in pid
log = root / 'logs/train_adares_y50_joint_incidence_res_20261016_145628.log'
with log.open('rb') as stream:
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(max(0, size - 8388608))
    text = stream.read(8388608).decode('utf-8', errors='replace')
progress = re.findall(r'Train\[(\d+)/(\d+)\]:[^\r\n]*?(\d+)/(\d+)\s*\[', text[-16384:])
assert len(progress) >= 2
assert tuple(map(int, progress[-1][:3])) >= tuple(map(int, progress[-2][:3]))
metric_lines = [line[:600] for line in re.split(r'[\r\n]', text) if any(marker in line for marker in ('mIoU :', 'mIOU:', 'Evaluate referring segmentation:', 'save best pth', 'save last pth'))][-12:]
uuids = {int(line.split(',')[0]): line.split(',')[1].strip() for line in gpu}
occupied = {i: any(uuid in line for line in processes) for i, uuid in uuids.items()}
assert set(occupied) == {0, 1} and all(occupied.values()), 'Resource block is not supported if a GPU became available'
assert any('19723' in line and uuids[0] in line for line in processes)
assert any('31539' in line and uuids[1] in line for line in processes)
result = dict(complete=True, project='AdaRES_project', no_GPU=True, read_only=True,
    resources=dict(gpu=gpu, compute_processes=processes, all_authorized_slots_occupied=occupied),
    Y50=dict(pid=31539, process=pid, native_progress=progress[-2:], metric_lines=metric_lines,
             crash_marker_in_last_16384_bytes=any(s in text[-16384:] for s in ('Traceback (most recent call last)', 'CUDA out of memory'))),
    block='exclusive_gpu_slot', authorized_gpu0_owner='foreign PID19723: preserved, not an AdaRES experiment',
    authorized_gpu1_owner='same existing AdaRES-Y50 PID31539: preserved',
    timestamps='Use client runner observation window; server clock is not trusted.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
