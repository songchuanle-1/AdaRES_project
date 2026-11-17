"""Exact original Y80 GPU0 run and retained additive Y81/Y82 sources."""
from y82_deployment import Y80, Y81, Y82, contract, git, validate_delta, paths

CID = 'AdaRES-Y80'
CHECKPOINT_DIRECTORY = '20261102_175917'
CONFIG = 'configs_song/model_paper5/adares_y80_query_visual_feedback_res.yaml'


def validate_marker(marker, candidate):
    assert candidate == CID and marker in (Y80, Y81, Y82)


def build_contract(task, server):
    assert task['id'] == CID and task['status'] == 'running' and task['launch_verified'] is True
    assert task['server'] == 'server-2' and str(task['gpu']) == '0' and task['pid'] == 42226
    assert task['code_commit'] == Y80 and task['config'] == CONFIG
    assert task['log'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y80_query_visual_feedback_res_20261103_070003.log'
    assert server['host'] == '192.168.1.163' and server['expected_hostname'] == 'dl01-MS-7C60'
    assert server['allowed_gpus'] == [0, 1] and server['max_tasks_per_gpu'] == 1
    assert server['python'] == '/home/songcl/.conda/envs/adares_project/bin/python'
    assert server['remote_code_root'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES'
    assert server['work_dir'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints'
    assert server['data_root'].rstrip('/') == '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data'
    assert not server.get('runtime_path_mappings')
    head = git('rev-parse', 'HEAD').decode().strip()
    versions = (Y80, Y81, Y82)
    assert head in versions
    deployments = contract(Y80, head, CID)
    assert set(deployments) == set(versions[:versions.index(head) + 1])
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
