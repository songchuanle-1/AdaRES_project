"""Add exactly the seven frozen Y49 files; preserve existing experiments."""
import fcntl,hashlib,json,subprocess,tarfile
from pathlib import Path
assert P['physical_gpu'] is None
root=Path(P['root'])
project=root.parent
assert str(root)=='/bak_home/songcl/Project/AdaRES_project/AdaRES'
lease=open(str(project/'.y15_gpu1_diagnostic.lock'),'a')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
used,util=map(int,subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().split(','))
assert used<1024 and util<10
assert not subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
marker=root/'.codex_source_commit'
assert marker.read_text().strip()=='d18c228fd3796b5a87dbe313075b6ee748b2d51b'
manifest=json.loads(Path('/tmp/adares_y50_bundle_1cb1.json').read_text())
assert manifest['complete'] and len(manifest['files'])==7
assert not (root/'ipdn/model_paper5/adares_y50_joint_incidence').exists()
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for n in ('adares_w2_pointmass_competition','adares_y45_metric_decomposition','adares_y48_mask_equivalence','adares_y46_component_coherence') for p in (root/'ipdn/model_paper5'/n).glob('*.py')}
archive=tarfile.open('/tmp/adares_y50_bundle_1cb1.tar','r:')
members=archive.getmembers()
assert sorted(m.name for m in members)==sorted(r['path'] for r in manifest['files'])
for m in members:
    assert m.isfile()
    assert m.name.startswith('AdaRES/ipdn/model_paper5/adares_y50_joint_incidence/') or m.name=='AdaRES/configs_song/model_paper5/adares_y50_joint_incidence_res.yaml'
    target=(project/m.name).resolve()
    assert project.resolve() in target.parents and not target.exists()
    content=archive.extractfile(m).read()
    assert hashlib.sha256(content).hexdigest()==next(r['sha256'] for r in manifest['files'] if r['path']==m.name)
    if m.name.endswith('.py'):compile(content.decode('utf-8'),m.name,'exec')
for m in members:
    target=project/m.name
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(archive.extractfile(m).read())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
assert marker.read_text().strip()=='d18c228fd3796b5a87dbe313075b6ee748b2d51b'
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y50',implementation_revision=1,files=manifest['files'],protected_files=len(protected),protected_sources_unchanged=True,marker_unchanged=True,no_training=True)),encoding='utf-8')
