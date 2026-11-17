"""Exact original Y85 GPU0 run; no replacement source or checkpoint directory."""
from y85_deployment import Y85, contract, git, paths, validate_delta
from y82_terminal_contract import IDENTITY

CID = 'AdaRES-Y85'
CHECKPOINT_DIRECTORY = '20261104_160627'
CONFIG = 'configs_song/model_paper5/adares_y85_mask_feature_norm_res.yaml'


def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y85


def build_contract(task, server):
    assert task['id'] == CID and task['status'] == 'running' and task['launch_verified'] is True
    assert (task['server'], str(task['gpu']), task['pid']) == ('server-2', '0', 22090)
    assert task['code_commit'] == Y85 and task['config'] == CONFIG
    assert task['log'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y85_mask_feature_norm_res_20261105_050720.log'
    assert server['host'] == '192.168.1.163' and server['expected_hostname'] == 'dl01-MS-7C60'
    assert server['allowed_gpus'] == [0] and server['max_tasks_per_gpu'] == 1
    assert server['python'] == '/home/songcl/.conda/envs/adares_project/bin/python'
    assert server['remote_code_root'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES'
    assert server['work_dir'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints'
    assert not server.get('runtime_path_mappings')
    head = git('rev-parse', 'HEAD').decode().strip()
    assert head == Y85
    deployments = contract(Y85, head, CID)
    assert set(deployments) == {Y85}
    return dict(allowed_markers=list(deployments), deployments=deployments)
