"""Bind only actual-native-tested Y101 additions to the original Y100 lifecycle."""
import copy
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT, ref, tick
from y100_terminal_contract import SPEC, CID, CHECKPOINT_DIRECTORY, CONFIG, IDENTITY
from y101_deployment import Y100, Y101, adapt, contract, git
from advance_y101 import prepared
from y101_contract import validate_result
import closure

def validate_marker(marker, candidate):
    assert candidate == CID and marker in (Y100, Y101)

def native_evidence(work):
    candidate = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y101'))
    assert candidate.get('native_smoke_result'), 'Actual additive Y101 native evidence required'
    candidate, _, _, frozen = prepared(work)
    entry = copy.deepcopy(candidate['native_smoke_result'])
    validate_result(tick.load(closure.artifact(ROOT, entry)), candidate, frozen)
    return entry

def build_contract(task, server):
    assert task['id'] == CID and task['status'] == 'running'
    assert task['launch_verified'] is True and str(task['gpu']) == '0'
    assert all((task[key] == SPEC[key] for key in ('server', 'pid', 'code_commit', 'config', 'log')))
    validate_server(server)
    assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
    assert ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'] == '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
    location = tick.load(closure.artifact(ROOT, SPEC['location_evidence']))
    assert location['complete'] is True and location['checkpoint_loaded'] is False
    assert location['remote_mutations'] is False and location['log'] == SPEC['log']
    assert location['checkpoint_directory'] == CHECKPOINT_DIRECTORY
    expected = PurePosixPath(server['work_dir']) / PurePosixPath(CONFIG).stem / CHECKPOINT_DIRECTORY / 'last.pth'
    assert PurePosixPath(location['declared_last_path']) == expected
    assert git('rev-parse', 'HEAD').decode().strip() in (Y100, Y101)
    assert git('rev-parse', Y101 + '^').decode().strip() == Y100
    native_evidence(tick.load(ROOT / 'pipeline/research_work.json'))
    deployments = contract(Y100, Y101, CID)
    assert list(deployments) == [Y100, Y101]
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)

def validate_server(server):
    closure.validate_adares_server(ROOT,server)
    for k,v in dict(host='192.168.1.163',expected_hostname='dl01-MS-7C60',python='/home/songcl/.conda/envs/adares_project/bin/python',remote_code_root='/bak_home/songcl/Project/AdaRES_project/AdaRES',work_dir='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints',allowed_gpus=[0],max_tasks_per_gpu=1).items():assert server[k]==v
    assert not server.get('runtime_path_mappings') and not server.get('runtime_library_path')
