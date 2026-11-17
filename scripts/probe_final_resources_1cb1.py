"""Read only the two authorized GPUs after diagnostics, preserve foreign jobs."""
import json
from pathlib import Path
import subprocess
assert P['physical_gpu'] is None
rows=[]
for gpu in (0,1):
    values=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
    rows.append(dict(gpu=gpu,memory_MiB=int(values[0]),total_MiB=int(values[1]),utilization=int(values[2]),pids=[int(v.strip()) for v in raw.splitlines() if v.strip()]))
Path(P['output']).write_text(json.dumps(dict(complete=True,authorized_GPU_rows=rows,read_only=True,training_launched=False)),encoding='utf-8')
