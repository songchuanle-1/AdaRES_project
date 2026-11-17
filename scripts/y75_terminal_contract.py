"""Original server2 Y75 identity under exact Y76/Y77 additive deployments."""
from y77_deployment import Y75,Y76,Y77,contract,git,validate_delta,paths

def validate_marker(marker,candidate):
    assert candidate=='AdaRES-Y75' and marker in (Y75,Y76,Y77)

def build_contract(task,server):
    assert task['id']=='AdaRES-Y75' and task['status']=='running' and task['launch_verified'] is True
    assert task['server']=='server-2' and str(task['gpu'])=='1' and task['pid']==65454
    assert task['code_commit']==Y75 and task['config']=='configs_song/model_paper5/adares_y75_language_query_init_res.yaml'
    assert server['host']=='192.168.1.163' and server['expected_hostname']=='dl01-MS-7C60'
    assert server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
    head=git('rev-parse','HEAD').decode().strip();assert head in (Y76,Y77)
    deployments=contract(Y75,head,'AdaRES-Y75')
    assert set(deployments)==({Y75,Y76} if head==Y76 else {Y75,Y76,Y77})
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
