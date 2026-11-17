"""Exact original Y68 identity and bounded local/preproduction source edges."""
import hashlib
import subprocess
import yaml
from prepare_y60_stage_heads import ROOT,ref
import closure
from observe_y68_with_local_y71 import validate_local_y71
from admit_dispatch_y70_gpu1 import validate_additive_delta

OLD='9af4deca98f121cbbe88e65ac7e233035f76b888'
PRODUCTION='39bdb8d9783512e70cdc2763e9a6097ab0f67a0f'
LOCAL='87019e7c2f640f99cb82a3e5b396321fc5b54d25'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')
IDENTITIES={'AdaRES-Y68':dict(gpu=0,pid=20742,code_commit=OLD,package='adares_y68_language_mask_modulation',watts=280.0,
    receipt_sha256='bc141d239288ae515b74ccf6869018626a6fed7141a135708012b95027a0f327')}

def validate_identity(task):
    identity=IDENTITIES[task['id']]
    assert task['status']=='running' and task['launch_verified'] is True
    assert str(task['gpu'])=='0' and task['pid']==identity['pid'] and task['code_commit']==OLD
    assert task['server']=='server-2' and task['config']=='configs_song/model_paper5/'+identity['package']+'_res.yaml'
    return identity

def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)

def build_contract(task,server):
    validate_identity(task)
    assert server['host']=='192.168.1.163' and server['expected_hostname']=='dl01-MS-7C60'
    assert server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
    assert closure.git(ROOT,'rev-parse','HEAD')==LOCAL and closure.git(ROOT,'rev-parse',LOCAL+'^')==PRODUCTION
    rows=subprocess.check_output(['git','diff','--no-renames','--name-status',PRODUCTION,LOCAL],cwd=ROOT).decode().splitlines()
    rows=[r.split('\t') for r in rows];validate_local_y71(rows)
    for status,path in rows:assert (ROOT/path).read_bytes()==blob(LOCAL,path)
    rows=subprocess.check_output(['git','diff','--no-renames','--name-status',OLD,PRODUCTION],cwd=ROOT).decode().splitlines()
    rows=[r.split('\t') for r in rows];validate_additive_delta(rows)
    additions=dict(source_files={},configs={})
    for status,path in rows:
        raw=blob(PRODUCTION,path);assert (ROOT/path).read_bytes()==raw
        relative=path[len('AdaRES/'):]
        if path.endswith('.yaml'):additions['configs'][relative]=yaml.safe_load(raw)
        else:additions['source_files'][relative]=hashlib.sha256(raw).hexdigest()
    for name in NAMES:
        path='AdaRES/ipdn/model_paper5/adares_y68_language_mask_modulation/'+name
        assert (ROOT/path).read_bytes()==blob(OLD,path)==blob(PRODUCTION,path)
    path='AdaRES/'+task['config']
    assert yaml.safe_load((ROOT/path).read_bytes())==yaml.safe_load(blob(OLD,path))==yaml.safe_load(blob(PRODUCTION,path))
    return dict(allowed_markers=[OLD,PRODUCTION],deployments={OLD:dict(source_files={},configs={}),PRODUCTION:additions})
