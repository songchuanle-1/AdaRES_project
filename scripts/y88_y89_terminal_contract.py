"""New additive Y89 binding for the unchanged original Y88 terminal contract."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT
from y88_terminal_contract import SPEC, CHECKPOINT_DIRECTORY, CONFIG, IDENTITY, paths, validate_delta
from current_terminal_contract import RUNS
from y89_deployment import Y88, Y89, contract, git, adapt
import closure


def validate_marker(marker, cid):
    assert cid == 'AdaRES-Y88' and marker in (Y88, Y89)


def build_contract(task, server):
    cid = 'AdaRES-Y88'
    expected = RUNS[cid]
    assert SPEC['id'] == cid and all(SPEC[key] == value for key, value in expected.items())
    assert task['id'] == cid and task['status'] == 'running' and task['launch_verified'] is True
    assert str(task['gpu']) == '0'
    assert all(task[key] == expected[key] for key in ('server', 'pid', 'code_commit', 'config', 'log'))
    closure.validate_adares_server(ROOT, server)
    assert server['allowed_gpus'] == [0] and server['max_tasks_per_gpu'] == 1
    for key, value in dict(host='192.168.1.175', expected_hostname='dl02-MS-7B73',
            python='/data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python',
            runtime_library_path='/data1/songcl/Project/AdaRES_project/runtime/cuda113lib64',
            remote_code_root='/data1/songcl/Project/AdaRES_project/AdaRES',
            work_dir='/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints').items():
        assert server[key] == value
    assert server['runtime_path_mappings'] == [
        dict(source='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data',
             target='/data1/songcl/Project/AdaRES_project/data'),
        dict(source='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints',
             target='/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints')]
    location = json.loads(closure.artifact(ROOT, SPEC['location_evidence']).read_text(encoding='utf-8'))
    assert location['checkpoint_loaded'] is False and location['remote_mutations'] is False
    assert location['log'] == expected['log'] and location['checkpoint_directory'] == CHECKPOINT_DIRECTORY
    target = PurePosixPath(server['work_dir']) / PurePosixPath(expected['config']).stem / CHECKPOINT_DIRECTORY / 'last.pth'
    assert PurePosixPath(location['declared_last_path']) == target
    assert git('rev-parse', 'HEAD').decode().strip() == Y89
    deployments = contract(Y88, Y89, cid)
    assert set(deployments) == {Y88, Y89}
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
