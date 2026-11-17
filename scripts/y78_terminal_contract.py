"""Exact original Y78 identity under registered Y79/Y80 additive versions."""
from y80_deployment import Y78,Y79,Y80,contract,git,validate_delta,paths


def validate_marker(marker,candidate):
    assert candidate=='AdaRES-Y78' and marker in (Y78,Y79,Y80)


def build_contract(task,server):
    assert task['id']=='AdaRES-Y78' and task['status']=='running' and task['launch_verified'] is True
    assert task['server']=='server-2' and str(task['gpu'])=='1' and task['pid']==22192
    assert task['code_commit']==Y78 and task['config']=='configs_song/model_paper5/adares_y78_mask_logit_residual_res.yaml'
    assert server['host']=='192.168.1.163' and server['expected_hostname']=='dl01-MS-7C60'
    assert server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
    assert server['python']=='/home/songcl/.conda/envs/adares_project/bin/python'
    assert server['data_root']=='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/'
    head=git('rev-parse','HEAD').decode().strip();assert head in (Y78,Y79,Y80)
    deployments=contract(Y78,head,'AdaRES-Y78');versions=(Y78,Y79,Y80)
    assert set(deployments)==set(versions[:versions.index(head)+1])
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
