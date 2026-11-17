"""Independent bounded log/process/GPU observation of the unique admitted run."""
import hashlib,json,os,subprocess,time
from pathlib import Path
assert P['physical_gpu'] is None
root=Path(P['root'])
pid=31539
config='configs_song/model_paper5/adares_y50_joint_incidence_res.yaml'
log=root/'logs/train_adares_y50_joint_incidence_res_20261016_145628.log'
args=Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\0')
assert b'tools/train.py' in args and config.encode() in args
assert args[args.index(b'--gpu_ids')+1]==b'1' and args[args.index(b'--num_gpus')+1]==b'1'
assert (root/'.codex_source_commit').read_text().strip()=='26ee1cec57c53ee2f94d6969d0bb637565b6ed59'
manifest=json.loads(Path('/tmp/adares_y50_bundle_1cb1.json').read_text())
assert all(hashlib.sha256((root.parent/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['files'])
samples=[]
for _ in range(2):
    os.kill(pid,0)
    with log.open('rb') as stream:
        size=os.fstat(stream.fileno()).st_size
        stream.seek(max(0,size-24000))
        text=stream.read().decode('utf-8',errors='replace')
    assert not any(word in text for word in ('Traceback','CUDA out of memory','RuntimeError'))
    samples.append(dict(log_bytes=size,tail=text))
    if len(samples)==1:time.sleep(3)
gpus=[]
for gpu in (0,1):
    values=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
    gpus.append(dict(gpu=gpu,memory_MiB=int(values[0]),total_MiB=int(values[1]),utilization=int(values[2]),pids=[int(v.strip()) for v in raw.splitlines() if v.strip()]))
assert pid in gpus[1]['pids'] and pid not in gpus[0]['pids']
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y50',pid=pid,config=config,log=str(log),
    exact_source_commit='26ee1cec57c53ee2f94d6969d0bb637565b6ed59',exact_tested_files=True,
    actual_single_GPU1=True,samples=samples,authorized_GPU_rows=gpus,read_only=True)),encoding='utf-8')
