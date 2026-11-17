"""Specialize proven lifecycle helpers additively, keeping all old files intact."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from current_terminal_contract import RUNS, IDENTITY
from y88_deployment import Y88, contract, paths, adapt, git
from finalize_y61_fixed30 import DISCOVER
import closure

SCRIPTS = ROOT / 'scripts'
LIVE_IDENTITY = '''
process=pathlib.Path('/proc')/str(P['pid'])
assert process.is_dir()
assert (process/'cwd').resolve()==pathlib.Path(P['root'])
command=(process/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert 'tools/train.py' in command and P['config'] in command
'''


def discover(token, cid, expected, server, task):
    deployments = contract(expected['code_commit'], Y88, cid)
    for deployment in deployments.values():
        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
    config = __import__('yaml').safe_load((ROOT / 'AdaRES' / expected['config']).read_text(encoding='utf-8'))
    payload = dict(id=cid, pid=expected['pid'], root=server['remote_code_root'], work=server['work_dir'],
        config=expected['config'], log=expected['log'], basename=Path(expected['config']).stem,
        code_commit=expected['code_commit'], expected_config=adapt(config, server),
        source_files={path[len('AdaRES/'):]: ref(path)['sha256'] for path in paths(cid)[:-1]},
        deployments=deployments, allowed_markers=list(deployments))
    body = DISCOVER.replace('dl01-MS-7C60', server['expected_hostname'])
    body = body.replace('print(json.dumps(dict(complete=True', IDENTITY + LIVE_IDENTITY + 'print(json.dumps(dict(complete=True')
    program = 'P=' + repr(payload) + '\n' + body
    ast.parse(program, feature_version=(3, 8))
    base = ROOT / ('pipeline/checks/' + cid.lower().replace('adares-', '') + '_terminal_original_location_' + token)
    output, source, log = base.with_suffix('.json'), base.with_suffix('.py'), base.with_suffix('.log')
    assert not any(path.exists() for path in (output, source, log))
    source.write_text(program, encoding='utf-8', newline='\n')
    command = 'PYTHONDONTWRITEBYTECODE=1 '
    if server.get('runtime_library_path'):
        command += 'LD_LIBRARY_PATH=' + shlex.quote(server['runtime_library_path']) + ' '
    command += shlex.quote(server['python']) + ' -B -'
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        server['user'] + '@' + server['host'], command], input=program.encode(),
        capture_output=True, timeout=45)
    log.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0, 'Preserve failed read-only discovery; never restart'
    value = json.loads(response.stdout)
    assert value['complete'] and value['checkpoint_loaded'] is False and value['remote_mutations'] is False
    assert value['log'] == expected['log']
    tick.save(output, value)
    return value['checkpoint_directory'], ref(output), [ref(source), ref(log)]


def specialize(text, old_id, new_id, values):
    text = text.replace(old_id, new_id).replace(old_id.lower(), new_id.lower())
    for old, new in values:
        assert old in text or old in ('243952', '22090')
        if old.isdecimal():
            text = re.sub(r'\b' + re.escape(old) + r'\b', new, text)
        else:
            text = text.replace(old, new)
    return text


def create(token):
    lease(token, 300)
    queue = monitor.load_queue(str(ROOT))
    running = monitor.load_running(str(ROOT))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    before_q, before_r = ref('pipeline/queue.json'), ref('pipeline/running.json')
    assert git('rev-parse', 'HEAD').decode().strip() == Y88
    targets = [SCRIPTS / name for suffix in ('87', '88') for name in (
        'finalize_y' + suffix + '_fixed30.py', 'prepare_y' + suffix + '_terminal_handoff.py',
        'test_y' + suffix + '_terminal_handoff.py', 'y' + suffix + '_terminal_contract.py')]
    assert not any(path.exists() for path in targets)
    originals = [ref(SCRIPTS / name) for suffix in ('84', '85') for name in (
        'finalize_y' + suffix + '_fixed30.py', 'prepare_y' + suffix + '_terminal_handoff.py',
        'test_y' + suffix + '_terminal_handoff.py', 'y' + suffix + '_terminal_contract.py')]
    discoveries, outputs = {}, {}
    for cid, expected in RUNS.items():
        server = monitor.load_servers(str(ROOT))[expected['server']]
        closure.validate_adares_server(ROOT, server)
        task = next(task for task in queue['gpu_queues'][expected['server']] if task['id'] == cid)
        candidate = next(c for c in work['candidates'] if c['id'] == cid)
        assert candidate['stage'] == 'training' and not candidate.get('terminal_handoff_plan')
        assert task['status'] == 'running' and task['launch_verified'] is True and str(task['gpu']) == '0'
        assert all(task[key] == expected[key] for key in ('pid', 'config', 'code_commit', 'log', 'server'))
        assert running[expected['server']]['0'] == cid
        assert ref('pipeline/admissions/' + cid + '.json')['sha256'] == expected['admission_sha256']
        assert candidate['post_training_freeze']['sha256'] == expected['freeze_sha256']
        closure.artifact(ROOT, candidate['post_training_freeze'])
        closure.admission(ROOT, task)
        directory, location, evidence = discover(token, cid, expected, server, task)
        spec = dict(expected, id=cid, checkpoint_directory=directory, location_evidence=location)
        discoveries[cid] = dict(spec=spec, evidence=evidence)
        suffix = cid[-3:].lower()
        contract_source = (
            '"""Exact original ' + cid + ' terminal source and checkpoint identity."""\n'
            'from current_terminal_contract import IDENTITY, Y87, Y88, paths, validate_delta\n'
            'from current_terminal_contract import build_contract as bound_contract, validate_marker\n\n'
            'SPEC = ' + repr(spec) + '\n'
            'CID = SPEC["id"]\nCHECKPOINT_DIRECTORY = SPEC["checkpoint_directory"]\nCONFIG = SPEC["config"]\n\n'
            'def build_contract(task, server):\n    return bound_contract(SPEC, task, server)\n')
        outputs[SCRIPTS / (suffix + '_terminal_contract.py')] = contract_source
        old = 'Y85' if cid == 'AdaRES-Y87' else 'Y84'
        replacements = [
            ('adares_' + cid[-3:].lower() + ('_mask_feature_norm' if old == 'Y85' else '_visual_query_init'),
             expected['package']),
            ('22090' if old == 'Y85' else '243952', str(expected['pid'])),
            ('20261104_160627' if old == 'Y85' else '20261105_032303', directory),
            ('20261105_050720' if old == 'Y85' else '20261105_032256',
             '20261106_032555' if cid == 'AdaRES-Y87' else '20261106_053534'),
            ('aa9d8db3ea05432181ada752eaeeca27952aae27' if old == 'Y85' else '7bef2cd313b870583c970c56580be5a78097234d',
             expected['code_commit']),
            ('c191a90dda4fdf0c2a1992ab5fca9d6164150c20025f6211cd2c3d9708f71c7a' if old == 'Y85'
             else 'c8ba93aa472ad4c54bd4af1904b5205a7426c7650445cae391cb603cca75f8aa', expected['admission_sha256']),
            ('4380dc32199f13af68e88833726f19399f3556d57446c2b8d62532298b4117bf' if old == 'Y85'
             else 'd3e38757764ede03fb84028fee837a8ed392d477fc32e0eb391d7676cb3b650c', expected['freeze_sha256']),
        ]
        for prefix in ('finalize_', 'prepare_', 'test_'):
            tail = '_fixed30.py' if prefix == 'finalize_' else '_terminal_handoff.py'
            origin = SCRIPTS / (prefix + old.lower() + tail)
            source = origin.read_text(encoding='utf-8')
            applicable = [pair for pair in replacements if pair[0] in source.replace(old.lower(), suffix).replace(old, cid[-3:])]
            text = specialize(source, old, cid[-3:], applicable)
            if prefix == 'prepare_':
                deployed = 'scripts/y87_deployment.py'
                assert deployed in text
                text = text.replace(deployed, "scripts/y88_deployment.py', 'scripts/current_terminal_contract.py")
            if prefix == 'test_' and cid == 'AdaRES-Y87':
                text = text.replace('import Y87,validate_marker', 'import Y87,Y88,validate_marker')
                text = text.replace('for value in (Y87,)', 'for value in (Y87,Y88)')
                text = text.replace("for cid in ('AdaRES-Y87',):", "for cid in ('AdaRES-Y88',):")
            if prefix == 'test_' and cid == 'AdaRES-Y88':
                text = text.replace('import Y88,Y85,Y87,validate_marker', 'import Y88,validate_marker')
                text = text.replace('for value in (Y88,Y85,Y87)', 'for value in (Y88,)')
                text = text.replace("for cid in ('AdaRES-Y85','AdaRES-Y87'):", "for cid in ('AdaRES-Y88',):")
            outputs[SCRIPTS / (prefix + suffix + tail)] = text
    assert set(outputs) == set(targets)
    for path, source in outputs.items():
        ast.parse(source, feature_version=(3, 8))
        assert not path.exists()
    for path, source in outputs.items():
        path.write_text(source, encoding='utf-8', newline='\n')
    assert [ref(ROOT / item['path']) for item in originals] == originals
    assert (before_q, before_r) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
    output = ROOT / ('pipeline/checks/y87_y88_terminal_helpers_' + token + '.json')
    assert not output.exists()
    tick.save(output, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
        originals=originals, discoveries=discoveries, generated=[ref(path) for path in outputs],
        dependencies=[ref(__file__), ref('scripts/current_terminal_contract.py')],
        source_specialization_only=True, original_helpers_unchanged=True,
        checkpoint_loaded=False, full_terminal_verification_executed=False,
        archived=False, queue_changed=False, training_stopped=False, remote_mutations=False))
    for cid in RUNS:
        candidate = next(c for c in work['candidates'] if c['id'] == cid)
        candidate['evidence'].append(ref(output))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    monitor.log_decision(str(ROOT), 'other', 'Y87/Y88在跑，后续候选原生检查需要原运行完整终态核验及归档。',
        '只新增绑定两个原始运行的终态检查/归档助手；本轮不加载checkpoint、不停止训练或改变队列。',
        '沿用已验证终态流程，精确绑定PID/日志/目录/收据/冻结协议与当前加法部署；完整校验仍须自然结束后执行。',
        output.relative_to(ROOT).as_posix())
    print(json.dumps(dict(generated=ref(output), checkpoint_directories={
        cid: value['spec']['checkpoint_directory'] for cid, value in discoveries.items()},
        checkpoint_loaded=False, archived=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        create(args.token)
    finally:
        writer.close()
