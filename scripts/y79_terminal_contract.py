"""Original Y79 server-1 run and retained additive Y80/Y81 production sources."""
from y81_deployment import Y79, Y80, Y81, contract, git, validate_delta, paths

CID = 'AdaRES-Y79'
CHECKPOINT_DIRECTORY = '20261102_230148'
CONFIG = 'configs_song/model_paper5/adares_y79_query_language_feedback_res.yaml'


def validate_marker(marker, candidate):
    assert candidate == CID and marker in (Y79, Y80, Y81)


def build_contract(task, server):
    assert task['id'] == CID and task['status'] == 'running' and task['launch_verified'] is True
    assert task['server'] == 'server-1' and str(task['gpu']) == '0' and task['pid'] == 2218231
    assert task['code_commit'] == Y79 and task['config'] == CONFIG
    assert task['log'] == '/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y79_query_language_feedback_res_20261102_230139.log'
    assert server['host'] == '192.168.1.175' and server['expected_hostname'] == 'dl02-MS-7B73'
    assert server['allowed_gpus'] == [0] and server['max_tasks_per_gpu'] == 1
    assert server['python'] == '/data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python'
    assert server['runtime_library_path'] == '/data1/songcl/Project/AdaRES_project/runtime/cuda113lib64'
    assert server['remote_code_root'] == '/data1/songcl/Project/AdaRES_project/AdaRES'
    assert server['work_dir'] == '/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints'
    assert server['runtime_path_mappings'] == [
        dict(source='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data', target='/data1/songcl/Project/AdaRES_project/data'),
        dict(source='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', target='/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints')]
    head = git('rev-parse', 'HEAD').decode().strip()
    versions = (Y79, Y80, Y81)
    assert head in versions
    deployments = contract(Y79, head, CID)
    assert set(deployments) == set(versions[:versions.index(head) + 1])

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
