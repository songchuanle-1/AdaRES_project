"""Exact original Y99 run; future source advances require a new tested binding."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT
from y84_terminal_contract import IDENTITY
from y99_deployment_v2 import Y99, adapt, contract, git, paths, validate_delta
import closure
SPEC = {'id': 'AdaRES-Y99', 'server': 'server-1', 'pid': 2980836, 'code_commit': '53f5b028bdf24e8b41dd934931d1fedc99843d8d', 'config': 'configs_song/model_paper5/adares_y99_spatial_mask_skip_res.yaml', 'log': '/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y99_spatial_mask_skip_res_20261111_180628.log', 'package': 'adares_y99_spatial_mask_skip', 'admission_sha256': '4d87ad8b3a6efe78be8dc9e8214bba9226160a22b649f64f8e1dd26d73988ede', 'freeze_sha256': 'fd07c0536891d905a7cd89e2b4211b32ed31a532b7b70ac9d71a8d7a66e97dbc', 'checkpoint_directory': '20261111_180638', 'location_evidence': {'path': 'pipeline/checks/y99_terminal_original_location_288dfa4d3bb741eaa67afacfdc81f241.json', 'sha256': '203cc4e1113d23ea01d1fbdfc7151cfda11cf4e3212aa04814caec4908381236'}}
MAPPINGS = [{'source': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data', 'target': '/data1/songcl/Project/AdaRES_project/data'}, {'source': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'target': '/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints'}]
CID = SPEC['id']
CHECKPOINT_DIRECTORY = SPEC['checkpoint_directory']
CONFIG = SPEC['config']

def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y99

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
    assert git('rev-parse', 'HEAD').decode().strip() == Y99
    deployments = contract(Y99, Y99, CID)
    assert set(deployments) == {Y99}
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
