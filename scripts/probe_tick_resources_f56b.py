"""Read authorized GPU occupation without allocating CUDA or touching jobs."""
import fcntl,json,subprocess
from pathlib import Path
assert P['physical_gpu'] is None
rows=[]
for gpu in (0,1):
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']).decode().strip()
    used,total,util=map(int,raw.split(','))
    pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
    rows.append(dict(gpu=gpu,memory_MiB=used,total_MiB=total,utilization=util,pids=[int(v) for v in pids.splitlines() if v.strip()]))
with open('/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock','rb') as lease:
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    diagnostic_lease_free=True
Path(P['output']).write_text(json.dumps(dict(complete=True,project='AdaRES_project',rows=rows,no_GPU_allocation=True,no_process_interference=True,server_clock_not_used=True,diagnostic_GPU1_lease_released=diagnostic_lease_free)),encoding='utf-8')
