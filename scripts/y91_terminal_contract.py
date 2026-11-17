"""Exact original Y91 terminal contract; future HEADs require a new tested binding."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT, ref
from y84_terminal_contract import IDENTITY
from y91_deployment import Y91, contract, git, paths, adapt, validate_delta
import closure

SPEC = {'id': 'AdaRES-Y91', 'server': 'server-2', 'pid': 83827, 'code_commit': '59baf94e9d3e406afa13b2ddc96968cdcc976cae', 'package': 'adares_y91_hidden_confidence_gate', 'config': 'configs_song/model_paper5/adares_y91_hidden_confidence_gate_res.yaml', 'log': '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y91_hidden_confidence_gate_res_20261107_232228.log', 'admission_sha256': '4d59825df7e4d0cfe428fc47c3ff6102749f2b1e5f12d84b71aaf5238aa96eaf', 'freeze_sha256': 'e5e6eef716ac892e827fe9131907ec44add8fdad1f619da8a568dacfc6935ca1', 'checkpoint_directory': '20261107_102124', 'location_evidence': {'path': 'pipeline/checks/y91_terminal_original_location_257d36d33c904645876d022a0fb35769.json', 'sha256': '87bf859d01ae86de3e5b4e7945ff18f1d04777f77bb03d53e7120a809bfe68ac'}}

CID = SPEC['id']
CHECKPOINT_DIRECTORY = SPEC['checkpoint_directory']
CONFIG = SPEC['config']


def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y91


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
    assert git('rev-parse', 'HEAD').decode().strip() == Y91
    deployments = contract(Y91, Y91, CID)
    assert set(deployments) == {Y91}
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
