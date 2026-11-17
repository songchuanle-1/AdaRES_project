"""Original Y71 completion identity under the exact Y72/Y73 additive edges."""
from prepare_y60_stage_heads import ROOT
from y72_y73_deployment import BASE,Y72,Y73,contract,git,validate_delta,paths,NAMES

def validate_marker(marker,candidate):
    assert candidate=='AdaRES-Y71' and marker in (BASE,Y72,Y73)

def build_contract(task,server):
    assert task['id']=='AdaRES-Y71' and task['status']=='running' and task['launch_verified'] is True
    assert task['server']=='server-2' and str(task['gpu'])=='0' and task['pid']==53714
    assert task['code_commit']==BASE and task['config']=='configs_song/model_paper5/adares_y71_visual_product_fusion_res.yaml'
    assert server['host']=='192.168.1.163' and server['expected_hostname']=='dl01-MS-7C60'
    assert server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
    assert git('rev-parse','HEAD').decode().strip()==Y73
    deployments=contract(BASE,Y73,'AdaRES-Y71')
    assert set(deployments)=={BASE,Y72,Y73}
    return dict(allowed_markers=[BASE,Y72,Y73],deployments=deployments)

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
