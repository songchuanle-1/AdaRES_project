"""Exact original Y93 terminal contract; future HEADs require a new tested binding."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT, ref
from y84_terminal_contract import IDENTITY
from y93_deployment import Y93, contract, git, paths, adapt, validate_delta
import closure

SPEC = {'id': 'AdaRES-Y93', 'server': 'server-2', 'pid': 47380, 'code_commit': 'daa93b4375f144f0e79da92f85fe1fe27e877e18', 'package': 'adares_y93_evidence_hidden_skip', 'config': 'configs_song/model_paper5/adares_y93_evidence_hidden_skip_res.yaml', 'log': '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y93_evidence_hidden_skip_res_20261108_212628.log', 'admission_sha256': 'f6b87f2813f1e8e429772bc0f63963fe1a43e0582c6c65e21771092dcbf69832', 'freeze_sha256': '8563a67a55b91f7c72b2899d03917b28d322e2f5ac14130c016511c72b0c7ef8', 'checkpoint_directory': '20261108_082521', 'location_evidence': {'path': 'pipeline/checks/y93_terminal_original_location_a7f1a4ec5ffc4c3a9837805bc45ce950.json', 'sha256': '454ba967c02b23f10722d70661cb829643dbeda62ed6042785122ad70edae517'}}

CID = SPEC['id']
CHECKPOINT_DIRECTORY = SPEC['checkpoint_directory']
CONFIG = SPEC['config']


def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y93


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
    assert git('rev-parse', 'HEAD').decode().strip() == Y93
    deployments = contract(Y93, Y93, CID)
    assert set(deployments) == {Y93}
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
