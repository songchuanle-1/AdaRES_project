"""Original Y73 completion identity under the exact registered Y74 additive edge."""
from y74_deployment import Y73,Y74,contract,git,validate_delta,paths

def validate_marker(marker,candidate):
    assert candidate=='AdaRES-Y73' and marker in (Y73,Y74)

def build_contract(task,server):
    assert task['id']=='AdaRES-Y73' and task['status']=='running' and task['launch_verified'] is True
    assert task['server']=='server-2' and str(task['gpu'])=='1' and task['pid']==62829
    assert task['code_commit']==Y73 and task['config']=='configs_song/model_paper5/adares_y73_initial_query_anchor_res.yaml'
    assert server['host']=='192.168.1.163' and server['expected_hostname']=='dl01-MS-7C60'
    assert server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
    assert git('rev-parse','HEAD').decode().strip()==Y74
    deployments=contract(Y73,Y74,'AdaRES-Y73')
    assert set(deployments)=={Y73,Y74}
    return dict(allowed_markers=[Y73,Y74],deployments=deployments)

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
