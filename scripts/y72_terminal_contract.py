"""Original server1 Y72 identity under explicitly registered Y73/Y74/Y75 additions."""
from y75_deployment import Y72,Y73,Y74,Y75,contract,git,validate_delta,paths

def validate_marker(marker,candidate):
    assert candidate=='AdaRES-Y72' and marker in (Y72,Y73,Y74,Y75)

def build_contract(task,server):
    assert task['id']=='AdaRES-Y72' and task['status']=='running' and task['launch_verified'] is True
    assert task['server']=='server-1' and str(task['gpu'])=='0' and task['pid']==311445
    assert task['code_commit']==Y72 and task['config']=='configs_song/model_paper5/adares_y72_visual_mask_refinement_res.yaml'
    assert server['host']=='192.168.1.175' and server['expected_hostname']=='dl02-MS-7B73'
    assert server['allowed_gpus']==[0] and server['max_tasks_per_gpu']==1
    head=git('rev-parse','HEAD').decode().strip();assert head in (Y74,Y75)
    deployments=contract(Y72,head,'AdaRES-Y72')
    assert set(deployments)==({Y72,Y73,Y74} if head==Y74 else {Y72,Y73,Y74,Y75})
    def adapt(v):
        if isinstance(v,dict):return {k:adapt(x) for k,x in v.items()}
        if isinstance(v,list):return [adapt(x) for x in v]
        if isinstance(v,str):
            for mapping in server['runtime_path_mappings']:
                old,new=mapping['source'].rstrip('/'),mapping['target'].rstrip('/')
                if v==old or v.startswith(old+'/'):v=new+v[len(old):]
        return v
    for d in deployments.values():d['configs']={p:adapt(v) for p,v in d['configs'].items()}
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
