"""Exact original Y95 terminal contract; future HEADs require a new tested binding."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT, ref
from y84_terminal_contract import IDENTITY
from y95_deployment import Y95, contract, git, paths, adapt, validate_delta
import closure

SPEC = {'id': 'AdaRES-Y95', 'server': 'server-2', 'pid': 34484, 'code_commit': '968112d44c506dcdd8dcf7709af997fb731e73b0', 'package': 'adares_y95_query_set_mean', 'config': 'configs_song/model_paper5/adares_y95_query_set_mean_res.yaml', 'log': '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y95_query_set_mean_res_20261109_212023.log', 'admission_sha256': 'de4e2620341bcd651a6d2ad3d8bfa98dc7e2e8523a7da9f1ffaea8c11d7597c5', 'freeze_sha256': '2248202035aaa859a4cab6d5b1e4f2ffd947f4775725390562f36a944ceed6c9', 'checkpoint_directory': '20261109_081912', 'location_evidence': {'path': 'pipeline/checks/y95_terminal_original_location_4380bd58933d48daa5860115c763a934.json', 'sha256': 'f27aa2ff9c872d94b5ef92147ea10e232dd6fde4263808ec26e29166a56fd7d5'}}

CID = SPEC['id']
CHECKPOINT_DIRECTORY = SPEC['checkpoint_directory']
CONFIG = SPEC['config']


def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y95


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
    assert git('rev-parse', 'HEAD').decode().strip() == Y95
    deployments = contract(Y95, Y95, CID)
    assert set(deployments) == {Y95}
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
