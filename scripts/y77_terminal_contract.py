"""Exact original Y77 identity under explicitly retained Y78/Y79 additions."""
from y79_deployment import Y77,Y78,Y79,contract,git,validate_delta,paths


def validate_marker(marker,candidate):
    assert candidate=='AdaRES-Y77' and marker in (Y77,Y78,Y79)


def build_contract(task,server):
    assert task['id']=='AdaRES-Y77' and task['status']=='running' and task['launch_verified'] is True
    assert task['server']=='server-2' and str(task['gpu'])=='0' and task['pid']==111334
    assert task['code_commit']==Y77 and task['config']=='configs_song/model_paper5/adares_y77_geometry_query_init_res.yaml'
    assert server['host']=='192.168.1.163' and server['expected_hostname']=='dl01-MS-7C60'
    assert server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
    assert server['python']=='/home/songcl/.conda/envs/adares_project/bin/python'
    assert server['data_root']=='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/'
    head=git('rev-parse','HEAD').decode().strip();assert head in (Y77,Y78,Y79)
    deployments=contract(Y77,head,'AdaRES-Y77');versions=(Y77,Y78,Y79)
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
