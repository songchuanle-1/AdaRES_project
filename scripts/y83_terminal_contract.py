"""Exact original Y83 GPU0 lineage; later source may only add the tested Y84 files."""
from y84_deployment import Y83, Y84, contract, git, paths, validate_delta
from y82_terminal_contract import IDENTITY

CID = 'AdaRES-Y83'
CHECKPOINT_DIRECTORY = '20261103_175050'
CONFIG = 'configs_song/model_paper5/adares_y83_stage_query_norm_res.yaml'


def validate_marker(marker, candidate):
    assert candidate == CID and marker in (Y83, Y84)


def build_contract(task, server):
    assert task['id'] == CID and task['status'] == 'running' and task['launch_verified'] is True
    assert (task['server'], str(task['gpu']), task['pid']) == ('server-2', '0', 20849)
    assert task['code_commit'] == Y83 and task['config'] == CONFIG
    assert task['log'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y83_stage_query_norm_res_20261104_065139.log'
    assert server['host'] == '192.168.1.163' and server['expected_hostname'] == 'dl01-MS-7C60'
    assert server['allowed_gpus'] == [0] and server['max_tasks_per_gpu'] == 1
    assert server['python'] == '/home/songcl/.conda/envs/adares_project/bin/python'
    assert server['remote_code_root'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES'
    assert server['work_dir'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints'
    assert not server.get('runtime_path_mappings')
    head = git('rev-parse', 'HEAD').decode().strip()
    versions = (Y83, Y84)
    assert head in versions
    deployments = contract(Y83, head, CID)
    assert set(deployments) == set(versions[:versions.index(head) + 1])
    return dict(allowed_markers=list(deployments), deployments=deployments)
