"""Original server2 Y74 identity under exact Y75/Y76 additive deployments."""
from y76_deployment import Y74,Y75,Y76,contract,git,validate_delta,paths

def validate_marker(marker,candidate):
    assert candidate=='AdaRES-Y74' and marker in (Y74,Y75,Y76)

def build_contract(task,server):
    assert task['id']=='AdaRES-Y74' and task['status']=='running' and task['launch_verified'] is True
    assert task['server']=='server-2' and str(task['gpu'])=='0' and task['pid']==49891
    assert task['code_commit']==Y74 and task['config']=='configs_song/model_paper5/adares_y74_language_update_gate_res.yaml'
    assert server['host']=='192.168.1.163' and server['expected_hostname']=='dl01-MS-7C60'
    assert server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
    head=git('rev-parse','HEAD').decode().strip();assert head in (Y75,Y76)
    deployments=contract(Y74,head,'AdaRES-Y74')
    assert set(deployments)==({Y74,Y75} if head==Y75 else {Y74,Y75,Y76})
    return dict(allowed_markers=list(deployments),deployments=deployments)

IDENTITY='''import hashlib,yaml
root=pathlib.Path(P['root'])
for relative,digest in P['source_files'].items():
 assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
assert yaml.safe_load((root/P['config']).read_text())==P['expected_config']
marker=(root/'.codex_source_commit').read_text().strip()
assert marker in P['allowed_markers']
for relative,digest in P['deployments'][marker]['source_files'].items():
 assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
for relative,config in P['deployments'][marker]['configs'].items():
 assert yaml.safe_load((root/relative).read_text())==config
'''
