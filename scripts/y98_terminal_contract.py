"""Exact original Y98 terminal contract; future HEADs require a new tested binding."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT, ref
from y84_terminal_contract import IDENTITY
from y98_deployment import Y98, contract, git, paths, adapt, validate_delta
import closure

SPEC = {'id': 'AdaRES-Y98', 'server': 'server-2', 'pid': 25962, 'code_commit': '15c3481ff32456160b5a94588565f68f0d2d2c5b', 'package': 'adares_y98_appearance_mask_skip', 'config': 'configs_song/model_paper5/adares_y98_appearance_mask_skip_res.yaml', 'log': '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y98_appearance_mask_skip_res_20261110_210741.log', 'admission_sha256': 'dc1a8a6dd3afde694d255cceca30b1b16009e8249390aa706632e2a567136a62', 'freeze_sha256': 'bcd19eedca805c8491568b3f81cf9cefa8dc4b0def8f6a09dd87c58f5e29d51c', 'checkpoint_directory': '20261110_080627', 'location_evidence': {'path': 'pipeline/checks/y98_terminal_original_location_b8a83c307bf14e8e90795586f3ac165b.json', 'sha256': '1aad9fbcd48c4806c228edcaecc7405fa3ef78f4de81bd59bc280a7b9a45a436'}}

CID = SPEC['id']
CHECKPOINT_DIRECTORY = SPEC['checkpoint_directory']
CONFIG = SPEC['config']


def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y98


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
    assert git('rev-parse', 'HEAD').decode().strip() == Y98
    deployments = contract(Y98, Y98, CID)
    assert set(deployments) == {Y98}
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
