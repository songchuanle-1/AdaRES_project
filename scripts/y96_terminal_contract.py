"""Exact original Y96 run; future source advances require a new tested binding."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT
from y84_terminal_contract import IDENTITY
from y96_deployment import Y96, adapt, contract, git, paths, validate_delta
import closure
SPEC = {'id': 'AdaRES-Y96', 'server': 'server-1', 'pid': 3893460, 'code_commit': 'acb8c1769b7f3d66a92d287da8859a06a138b8b0', 'config': 'configs_song/model_paper5/adares_y96_seed_query_skip_res.yaml', 'log': '/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y96_seed_query_skip_res_20261110_165652.log', 'package': 'adares_y96_seed_query_skip', 'admission_sha256': '5dcd15c8da9efca80c530a5c61767fc68f9976a2e0c70aedcef3f5807ea42b1c', 'freeze_sha256': '50213e2f1fda98e5e9477eaa5c030952d1d913d3d1aeff86d1fc6aa984dfc154', 'checkpoint_directory': '20261110_165702', 'location_evidence': {'path': 'pipeline/checks/y96_terminal_original_location_2a7078533cd8470c9c2648b09bad94af.json', 'sha256': '1e428c2fedc61d52d3947cbf51fd046aebae8650ba6bfd31b13bac703c41772a'}}
MAPPINGS = [{'source': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data', 'target': '/data1/songcl/Project/AdaRES_project/data'}, {'source': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'target': '/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints'}]
CID = SPEC['id']
CHECKPOINT_DIRECTORY = SPEC['checkpoint_directory']
CONFIG = SPEC['config']

def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y96

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
    assert git('rev-parse', 'HEAD').decode().strip() == Y96
    deployments = contract(Y96, Y96, CID)
    assert set(deployments) == {Y96}
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
