"""Exact original Y76 identity and explicitly retained Y77/Y78 additive versions."""
from y78_deployment import Y76, Y77, Y78, contract, git, validate_delta, paths


def validate_marker(marker, candidate):
    assert candidate == 'AdaRES-Y76' and marker in (Y76, Y77, Y78)


def build_contract(task, server):
    assert task['id'] == 'AdaRES-Y76' and task['status'] == 'running' and task['launch_verified'] is True
    assert task['server'] == 'server-1' and str(task['gpu']) == '0' and task['pid'] == 3489002
    assert task['code_commit'] == Y76 and task['config'] == 'configs_song/model_paper5/adares_y76_stage_confidence_conditioning_res.yaml'
    assert server['host'] == '192.168.1.175' and server['expected_hostname'] == 'dl02-MS-7B73'
    assert server['allowed_gpus'] == [0] and server['max_tasks_per_gpu'] == 1
    assert server['python'] == '/data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python'
    assert server['runtime_library_path'] == '/data1/songcl/Project/AdaRES_project/runtime/cuda113lib64'
    head = git('rev-parse', 'HEAD').decode().strip()
    assert head in (Y76, Y77, Y78)
    deployments = contract(Y76, head, 'AdaRES-Y76')
    versions = (Y76, Y77, Y78)
    assert set(deployments) == set(versions[:versions.index(head)+1])
    def adapt(value):
        if isinstance(value, dict):
            return {key: adapt(item) for key, item in value.items()}
        if isinstance(value, list):
            return [adapt(item) for item in value]
        if isinstance(value, str):
            for mapping in server['runtime_path_mappings']:
                old, new = mapping['source'].rstrip('/'), mapping['target'].rstrip('/')
                if value == old or value.startswith(old + '/'):
                    value = new + value[len(old):]
        return value
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)


IDENTITY = '''import hashlib,yaml
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
