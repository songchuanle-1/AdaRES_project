"""Read-only exact-source and authorized-GPU checks before canonical dispatch."""
import hashlib,json,subprocess
from pathlib import Path
assert P['physical_gpu'] is None
root=Path(P['root'])
manifest_path=Path('/tmp/adares_y50_bundle_1cb1.json')
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=='2dfd88afafd55823f785f6399c4ebc3a46c2f8d5815d5f84eca6a16fa6e7530e'
manifest=json.loads(manifest_path.read_text())
rows=[]
for row in manifest['files']:
    path=root.parent/row['path']
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==row['sha256'],'Remote tested bytes changed: '+row['path']
    rows.append(dict(path=row['path'],sha256=actual))
parent=root/'ipdn/model_paper5/adares_w2_pointmass_competition'
candidate=root/'ipdn/model_paper5/adares_y50_joint_incidence'
for name in ('__init__.py','backbone.py','ipdn.py','loss.py','sample_model.py'):
    assert (candidate/name).read_bytes()==(parent/name).read_bytes()
assert hashlib.sha256((parent/'dec.py').read_bytes()).hexdigest()=='6362d882662b0e51594cd4af994359b93f99497affc93600322a34ca6c349b01'
gpus=[]
for gpu in (0,1):
    values=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
    gpus.append(dict(gpu=gpu,memory_MiB=int(values[0]),total_MiB=int(values[1]),utilization=int(values[2]),pids=[int(v.strip()) for v in raw.splitlines() if v.strip()]))
train=root/'tools/train.py'
lines=train.read_text().splitlines()
context=[dict(line=i+1,text=line) for i,line in enumerate(lines) if any(word in line for word in ('work_dir','basename','pretrain','cfg.config'))]
result=dict(complete=True,id='AdaRES-Y50',exact_remote_seven_files=True,files=rows,parent_bytes_unchanged=True,
    source_marker=(root/'.codex_source_commit').read_text().strip(),authorized_GPU_rows=gpus,
    train_entry_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),train_workdir_context=context,
    read_only=True,training_launched=False)
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
