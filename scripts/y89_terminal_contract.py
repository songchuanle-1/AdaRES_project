"""Exact original Y89 terminal contract on the tested additive Y90 source."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT
from y84_terminal_contract import IDENTITY
from y90_deployment import Y89, Y90, contract, git, paths, adapt, validate_delta
import closure

SPEC = {'id': 'AdaRES-Y89', 'server': 'server-2', 'pid': 60578, 'code_commit': '5f1c4bde4b682fabd84fdc84a4341ba2c845fe71', 'package': 'adares_y89_stage_evidence_norm', 'config': 'configs_song/model_paper5/adares_y89_stage_evidence_norm_res.yaml', 'log': '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y89_stage_evidence_norm_res_20261107_012457.log', 'admission_sha256': '0ebbe5d0997ad2a3ac8e2d89ba11cbb9e5d47bd0b5e3710de9aa24a06d0220ec', 'freeze_sha256': '4ea48fe7d44504047205956a929b32dc18a094319f975707ca5f806c54d2347c', 'checkpoint_directory': '20261106_122357', 'location_evidence': {'path': 'pipeline/checks/y89_terminal_original_location_4dc4c79601f6427b82195a7dac4bc90c.json', 'sha256': '1d357ccdee7a4af0f89b4ae92d5b32060e4c4df9a56b735ee15b19e2f272f44a'}}

CID = SPEC['id']
CHECKPOINT_DIRECTORY = SPEC['checkpoint_directory']
CONFIG = SPEC['config']


def validate_marker(marker, candidate):
    assert candidate == CID and marker in (Y89, Y90)


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
    location = json.loads(closure.artifact(ROOT, SPEC['location_evidence']).read_text(encoding='utf-8'))
    assert location['checkpoint_loaded'] is False and location['remote_mutations'] is False
    assert location['log'] == SPEC['log'] and location['checkpoint_directory'] == CHECKPOINT_DIRECTORY
    expected = PurePosixPath(server['work_dir']) / PurePosixPath(CONFIG).stem / CHECKPOINT_DIRECTORY / 'last.pth'
    assert PurePosixPath(location['declared_last_path']) == expected
    assert git('rev-parse', 'HEAD').decode().strip() == Y90
    deployments = contract(Y89, Y90, CID)
    assert set(deployments) == {Y89, Y90}
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
