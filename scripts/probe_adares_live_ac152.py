"""Bounded read-only snapshots of the two exact existing AdaRES runs."""
import csv
import json
from pathlib import Path
import re
import subprocess
import time

assert P['physical_gpu'] is None
ROOT = Path(P['root'])
RUNS = [
    dict(id='AdaRES-Y52', pid=69064, gpu=1, config='configs_song/model_paper5/adares_y52_point_target_res.yaml',
         log='logs/train_adares_y52_point_target_res_20261019_164118.log'),
    dict(id='AdaRES-Y54', pid=47332, gpu=0, config='configs_song/model_paper5/adares_y54_fine_support_res.yaml',
         log='logs/train_adares_y54_fine_support_res_20261019_154936.log'),
]


def rows(command):
    return list(csv.reader(subprocess.check_output(command, timeout=10).decode().splitlines(), skipinitialspace=True))


def sample():
    gpu_rows = rows(['nvidia-smi', '-i', '0,1', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'])
    processes = rows(['nvidia-smi', '-i', '0,1', '--query-compute-apps=pid,gpu_uuid,used_memory',
                      '--format=csv,noheader,nounits'])
    gpus = [dict(index=int(i), uuid=uuid, utilization=int(util), memory_mib=int(mem),
                 pids=[int(p) for p, u, m in processes if u == uuid]) for i, uuid, util, mem in gpu_rows]
    result = []
    for spec in RUNS:
        proc = Path('/proc') / str(spec['pid'])
        argv = [item.decode() for item in (proc / 'cmdline').read_bytes().split(b'\0') if item]
        assert spec['config'] in argv and str(spec['gpu']) == argv[argv.index('--gpu_ids') + 1]
        assert argv[argv.index('--num_gpus') + 1] == '1'
        assert (proc / 'cwd').resolve() == ROOT
        assert spec['pid'] in next(row['pids'] for row in gpus if row['index'] == spec['gpu'])
        log = ROOT / spec['log']
        with log.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 4 * 1024 * 1024))
            text = stream.read(4 * 1024 * 1024).decode('utf-8', errors='replace')
        progress = re.findall(r'Train\[(\d+)/(\d+)\]:[^\r\n]*?(\d+)/(\d+)\s*\[', text[-16384:])
        result.append(dict(id=spec['id'], pid=spec['pid'], gpu=spec['gpu'], cwd_config_exact=True,
            log_size=size, last_step=[int(v) for v in progress[-1]] if progress else None,
            crash_marker_in_bounded_tail=any(marker in text[-16384:] for marker in ('Traceback (most recent call last)', 'CUDA out of memory')),
            current_attempt_stdout_rounded_miou=re.findall(r'mIoU\s*:\s*([0-9.]+)', text)[-4:]))
    return dict(GPUs=gpus, runs=result)


first = sample()
time.sleep(5)
second = sample()
increased = {}
for before, after in zip(first['runs'], second['runs']):
    a, b = before['last_step'], after['last_step']
    increased[after['id']] = a is not None and b is not None and (b[0], b[2]) > (a[0], a[2])
result = dict(complete=True, project='AdaRES_project', read_only=True, gpu_tests=0,
    first=first, second=second, native_steps_increased=increased,
    free_authorized_gpus=[row['index'] for row in second['GPUs']
                          if row['memory_mib'] < 1024 and row['utilization'] < 10 and not row['pids']],
    no_training_source_config_or_process_changes=True, training_launches=0,
    timestamp_rule='Runner local observation window; native server clock not trusted.',
    metric_rule='Only bounded CURRENT-attempt stdout rounded values; not new evaluation or precise historical metric.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
