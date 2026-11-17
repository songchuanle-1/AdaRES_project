"""Exact original Y100 terminal contract; future HEADs require a new tested binding."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT, ref
from y84_terminal_contract import IDENTITY
from y100_deployment import Y100, contract, git, paths, adapt, validate_delta
import closure

SPEC = {'id': 'AdaRES-Y100', 'server': 'server-2', 'pid': 19305, 'code_commit': '294de23f52e4ff93ebbf370c17c13d9a7d02e10b', 'package': 'adares_y100_appearance_input_skip', 'config': 'configs_song/model_paper5/adares_y100_appearance_input_skip_res.yaml', 'log': '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y100_appearance_input_skip_res_20261111_215359.log', 'admission_sha256': '7091f206c21429c02349c73e68d5e2b06ca3adc9f585465450870ac2a0e019f2', 'freeze_sha256': 'cb9b9a80fb6f9f9b116f9e9a35a515b3c4fb542eb6f4128cbee0ed5d389eff01', 'checkpoint_directory': '20261111_085241', 'location_evidence': {'path': 'pipeline/checks/y100_terminal_original_location_642f07d38b07499b853f2dc42e92e1d7.json', 'sha256': '816ef5a96fdb539ec5baa2572bb4b99dacce1660fede1290d838b4a121b8c178'}}

CID = SPEC['id']
CHECKPOINT_DIRECTORY = SPEC['checkpoint_directory']
CONFIG = SPEC['config']


def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y100


def build_contract(task, server):
    assert task['id'] == CID and task['status'] == 'running' and task['launch_verified'] is True
    assert str(task['gpu']) == '0'
    assert all(task[key] == SPEC[key] for key in ('server', 'pid', 'code_commit', 'config', 'log'))
    closure.validate_adares_server(ROOT, server)
    assert server['allowed_gpus'] == [0] and server['max_tasks_per_gpu'] == 1
    for key, value in dict(host='192.168.1.163', expected_hostname='dl01-MS-7C60',
            python='/home/songcl/.conda/envs/adares_project/bin/python',
            remote_code_root='/bak_home/songcl/Project/AdaRES_project/AdaRES',
            work_dir='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints').items():
        assert server[key] == value
    assert not server.get('runtime_path_mappings')
    assert ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'] == (
        '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')
    location = json.loads(closure.artifact(ROOT, SPEC['location_evidence']).read_text(encoding='utf-8'))
    assert location['checkpoint_loaded'] is False and location['remote_mutations'] is False
    assert location['log'] == SPEC['log'] and location['checkpoint_directory'] == CHECKPOINT_DIRECTORY
    expected = PurePosixPath(server['work_dir']) / PurePosixPath(CONFIG).stem / CHECKPOINT_DIRECTORY / 'last.pth'
    assert PurePosixPath(location['declared_last_path']) == expected
    assert git('rev-parse', 'HEAD').decode().strip() == Y100
    deployments = contract(Y100, Y100, CID)
    assert set(deployments) == {Y100}
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
