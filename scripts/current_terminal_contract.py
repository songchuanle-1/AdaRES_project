"""Bind Y87/Y88 terminal preparation to their exact original GPU0 runs."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT
from y88_deployment import Y87, Y88, contract, git, paths, adapt
from y84_terminal_contract import IDENTITY
import closure

RUNS = {
    'AdaRES-Y87': dict(server='server-2', pid=26281, code_commit=Y87,
        package='adares_y87_self_language_product',
        log='/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y87_self_language_product_res_20261106_032555.log',
        admission_sha256='300d46379e6072066f775c03ceaf2142d806a9fe02c3ad88f742126f5263e30c',
        freeze_sha256='b5aab97e1d694e64315cf0a0687a2c8c35a6eb5e73f5915cbffb9f36b9eaa3e4'),
    'AdaRES-Y88': dict(server='server-1', pid=3527278, code_commit=Y88,
        package='adares_y88_mask_query_quadratic',
        log='/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y88_mask_query_quadratic_res_20261106_053534.log',
        admission_sha256='39f9a71debe48314d479b42b604565cf557ccbe40e5582e949a66959587b3572',
        freeze_sha256='93bf51b65a580cbdd5a7ade69fae370d8d0e6e6db5ab665645892c775d84aa81'),
}
for _run in RUNS.values():
    _run['config'] = 'configs_song/model_paper5/' + _run['package'] + '_res.yaml'


def validate_marker(marker, candidate):
    assert candidate in RUNS
    assert marker in ((Y87, Y88) if candidate == 'AdaRES-Y87' else (Y88,))


def build_contract(spec, task, server):
    cid = spec['id']
    expected = RUNS[cid]
    assert all(spec[key] == value for key, value in expected.items())
    assert task['id'] == cid and task['status'] == 'running' and task['launch_verified'] is True
    assert str(task['gpu']) == '0'
    assert all(task[key] == expected[key] for key in ('server', 'pid', 'code_commit', 'config', 'log'))
    closure.validate_adares_server(ROOT, server)
    assert server['allowed_gpus'] == [0] and server['max_tasks_per_gpu'] == 1
    if expected['server'] == 'server-1':
        assert server['host'] == '192.168.1.175' and server['expected_hostname'] == 'dl02-MS-7B73'
        assert server['python'] == '/data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python'
        assert server['runtime_library_path'] == '/data1/songcl/Project/AdaRES_project/runtime/cuda113lib64'
        assert server['remote_code_root'] == '/data1/songcl/Project/AdaRES_project/AdaRES'
        assert server['work_dir'] == '/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints'
        assert server['runtime_path_mappings'] == [
            dict(source='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data',
                 target='/data1/songcl/Project/AdaRES_project/data'),
            dict(source='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints',
                 target='/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints')]
    else:
        assert server['host'] == '192.168.1.163' and server['expected_hostname'] == 'dl01-MS-7C60'
        assert server['python'] == '/home/songcl/.conda/envs/adares_project/bin/python'
        assert server['remote_code_root'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES'
        assert server['work_dir'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints'
        assert not server.get('runtime_path_mappings')
    location = json.loads(closure.artifact(ROOT, spec['location_evidence']).read_text(encoding='utf-8'))
    assert location['checkpoint_loaded'] is False and location['remote_mutations'] is False
    assert location['log'] == expected['log'] and location['checkpoint_directory'] == spec['checkpoint_directory']
    target = PurePosixPath(server['work_dir']) / PurePosixPath(expected['config']).stem / spec['checkpoint_directory'] / 'last.pth'
    assert PurePosixPath(location['declared_last_path']) == target
    head = git('rev-parse', 'HEAD').decode().strip()
    assert head == Y88
    deployments = contract(expected['code_commit'], head, cid)
    assert set(deployments) == ({Y87, Y88} if cid == 'AdaRES-Y87' else {Y88})
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)


def validate_delta(rows, cid):
    assert cid == 'AdaRES-Y88'
    assert len(rows) == 7 and all(len(row) == 2 and row[0] == 'A' for row in rows)
    assert {row[1] for row in rows} == set(paths(cid))
