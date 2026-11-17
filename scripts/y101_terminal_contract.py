"""Exact original Y101 run; future source advances require a new tested binding."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT
from y84_terminal_contract import IDENTITY
from y101_deployment import Y101, adapt, contract, git, paths, validate_delta
import closure
SPEC = {'id': 'AdaRES-Y101', 'server': 'server-1', 'pid': 2263581, 'code_commit': '6fb26984ae652efbba177ab20c39575c44a69517', 'config': 'configs_song/model_paper5/adares_y101_query_ffn_skip_res.yaml', 'log': '/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y101_query_ffn_skip_res_20261112_200229.log', 'package': 'adares_y101_query_ffn_skip', 'admission_sha256': '379a5b050624141f20345f614877f062f1a61309b55531fe977e023538fecd0c', 'freeze_sha256': 'e624a80505b35aa2fa09fa55bde64ed8f8e1915dbe385da3073db5b44bc81db1', 'checkpoint_directory': '20261112_200238', 'location_evidence': {'path': 'pipeline/checks/y101_terminal_original_location_473fa68cbdac4797ae9c836144db7e63.json', 'sha256': '7107a190fb4ca28ce7f8720dc8fa0d36675456f705033464dd80534089feca70'}}
MAPPINGS = [{'source': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data', 'target': '/data1/songcl/Project/AdaRES_project/data'}, {'source': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'target': '/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints'}]
CID = SPEC['id']
CHECKPOINT_DIRECTORY = SPEC['checkpoint_directory']
CONFIG = SPEC['config']

def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y101

def validate_server(server):
    closure.validate_adares_server(ROOT, server)
    assert server['allowed_gpus'] == [0] and server['max_tasks_per_gpu'] == 1
    for key, value in dict(host='192.168.1.175', expected_hostname='dl02-MS-7B73', python='/data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python', runtime_library_path='/data1/songcl/Project/AdaRES_project/runtime/cuda113lib64', remote_code_root='/data1/songcl/Project/AdaRES_project/AdaRES', work_dir='/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints').items():
        assert server[key] == value
    assert server['runtime_path_mappings'] == MAPPINGS

def build_contract(task, server):
    assert task['id'] == CID and task['status'] == 'running'
    assert task['launch_verified'] is True and str(task['gpu']) == '0'
    assert all((task[key] == SPEC[key] for key in ('server', 'pid', 'code_commit', 'config', 'log')))
    validate_server(server)
    location = json.loads(closure.artifact(ROOT, SPEC['location_evidence']).read_text(encoding='utf-8'))
    assert location['complete'] is True
    assert location['checkpoint_loaded'] is False and location['remote_mutations'] is False
    assert location['log'] == SPEC['log'] and location['checkpoint_directory'] == CHECKPOINT_DIRECTORY
    expected = PurePosixPath(server['work_dir']) / PurePosixPath(CONFIG).stem / CHECKPOINT_DIRECTORY / 'last.pth'
    assert PurePosixPath(location['declared_last_path']) == expected
    assert git('rev-parse', 'HEAD').decode().strip() == Y101
    deployments = contract(Y101, Y101, CID)
    assert set(deployments) == {Y101}
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
