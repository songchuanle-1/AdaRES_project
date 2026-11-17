"""Specialize established lifecycle guards for the exact still-running Y89."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finalize_y61_fixed30 import DISCOVER
from y84_terminal_contract import IDENTITY
from y90_deployment import Y89, Y90, contract, git, paths, adapt
import closure

CID = 'AdaRES-Y89'
SCRIPTS = ROOT / 'scripts'
SPEC = dict(id=CID, server='server-2', pid=60578, code_commit=Y89,
    package='adares_y89_stage_evidence_norm',
    config='configs_song/model_paper5/adares_y89_stage_evidence_norm_res.yaml',
    log='/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y89_stage_evidence_norm_res_20261107_012457.log',
    admission_sha256='0ebbe5d0997ad2a3ac8e2d89ba11cbb9e5d47bd0b5e3710de9aa24a06d0220ec',
    freeze_sha256='4ea48fe7d44504047205956a929b32dc18a094319f975707ca5f806c54d2347c')

LIVE_IDENTITY = '''
process=pathlib.Path('/proc')/str(P['pid'])
assert process.is_dir() and (process/'cwd').resolve()==pathlib.Path(P['root'])
args=(process/'cmdline').read_bytes().split(b'\\0')
assert b'tools/train.py' in args and P['config'].encode() in args
'''

CONTRACT = '''
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
'''


def create(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y90
        before = {path: ref(path) for path in ('pipeline/queue.json', 'pipeline/running.json',
                  'pipeline/goal.json', 'SERVERS.yaml')}
        index = git('ls-files', '--stage', '-z')
        queue = monitor.load_queue(str(ROOT))
        task = next(row for row in queue['gpu_queues']['server-2'] if row['id'] == CID)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(row for row in work['candidates'] if row['id'] == CID)
        assert candidate['stage'] == 'training' and not candidate.get('terminal_handoff_plan')
        assert task['status'] == 'running' and task['launch_verified'] is True and str(task['gpu']) == '0'
        assert all(task[key] == SPEC[key] for key in ('server', 'pid', 'code_commit', 'config', 'log'))
        assert monitor.load_running(str(ROOT))['server-2']['0'] == CID
        server = monitor.load_servers(str(ROOT))['server-2']
        closure.validate_adares_server(ROOT, server)
        closure.admission(ROOT, task)
        assert ref('pipeline/admissions/' + CID + '.json')['sha256'] == SPEC['admission_sha256']
        assert candidate['post_training_freeze']['sha256'] == SPEC['freeze_sha256']
        closure.artifact(ROOT, candidate['post_training_freeze'])
        outputs = {name: SCRIPTS / name for name in ('finalize_y89_fixed30.py',
            'test_y89_terminal_handoff.py', 'prepare_y89_terminal_handoff.py', 'y89_terminal_contract.py')}
        assert not any(path.exists() for path in outputs.values())
        originals = [ref('scripts/' + name) for name in ('finalize_y87_fixed30.py',
            'test_y87_terminal_handoff.py', 'prepare_y87_terminal_handoff.py', 'y87_terminal_contract.py')]
        deployments = contract(Y89, Y90, CID)
        for deployment in deployments.values():
            deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
        config = __import__('yaml').safe_load((ROOT / 'AdaRES' / SPEC['config']).read_bytes())
        payload = dict(id=CID, pid=SPEC['pid'], root=server['remote_code_root'], work=server['work_dir'],
            config=SPEC['config'], log=SPEC['log'], basename=Path(SPEC['config']).stem,
            code_commit=Y89, expected_config=adapt(config, server),
            source_files={path[len('AdaRES/'):]: ref(path)['sha256'] for path in paths(CID)[:-1]},
            deployments=deployments, allowed_markers=list(deployments))
        body = DISCOVER.replace('print(json.dumps(dict(complete=True',
            IDENTITY + LIVE_IDENTITY + 'print(json.dumps(dict(complete=True')
        program = 'P=' + repr(payload) + '\n' + body
        ast.parse(program, feature_version=(3, 8))
        base = ROOT / ('pipeline/checks/y89_terminal_original_location_' + token)
        location_path, source, log = base.with_suffix('.json'), base.with_suffix('.py'), base.with_suffix('.log')
        assert not any(path.exists() for path in (location_path, source, log))
        source.write_text(program, encoding='utf-8', newline='\n')
        response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            server['user'] + '@' + server['host'],
            'PYTHONDONTWRITEBYTECODE=1 ' + server['python'] + ' -B -'],
            input=program.encode(), capture_output=True, timeout=45)
        log.write_bytes(response.stdout + response.stderr)
        assert response.returncode == 0, 'Preserve failed read-only discovery; never restart'
        value = json.loads(response.stdout)
        assert value['complete'] is True and value['checkpoint_loaded'] is False and value['remote_mutations'] is False
        assert value['log'] == SPEC['log']
        tick.save(location_path, value)
        spec = dict(SPEC, checkpoint_directory=value['checkpoint_directory'], location_evidence=ref(location_path))
        generated = {}
        contract_source = ('"""Exact original Y89 terminal contract on the tested additive Y90 source."""\n'
            'import json\nfrom pathlib import PurePosixPath\nfrom prepare_y60_stage_heads import ROOT\n'
            'from y84_terminal_contract import IDENTITY\n'
            'from y90_deployment import Y89, Y90, contract, git, paths, adapt, validate_delta\n'
            'import closure\n\nSPEC = ' + repr(spec) + '\n' + CONTRACT)
        generated['y89_terminal_contract.py'] = contract_source
        replacements = [
            ('adares_y89_self_language_product', SPEC['package']),
            ('20261105_142459', spec['checkpoint_directory']),
            ('47ec0813cb4131ddf3e0a08db4fa9ac877123685', Y89),
            ('300d46379e6072066f775c03ceaf2142d806a9fe02c3ad88f742126f5263e30c', SPEC['admission_sha256']),
            ('b5aab97e1d694e64315cf0a0687a2c8c35a6eb5e73f5915cbffb9f36b9eaa3e4', SPEC['freeze_sha256'])]
        for prefix, suffix in (('finalize_', '_fixed30.py'), ('test_', '_terminal_handoff.py'),
                               ('prepare_', '_terminal_handoff.py')):
            original = SCRIPTS / (prefix + 'y87' + suffix)
            text = original.read_text(encoding='utf-8').replace('Y87', 'Y89').replace('y87', 'y89')
            text = re.sub(r'\b26281\b', '60578', text)
            for old, new in replacements:
                text = text.replace(old, new)
            if prefix == 'test_':
                text = text.replace('import Y89,Y88,validate_marker', 'import Y89,Y90,validate_marker')
                text = text.replace('for value in (Y89,Y88)', 'for value in (Y89,Y90)')
                text = text.replace("for cid in ('AdaRES-Y88',):", "for cid in ('AdaRES-Y90',):")
            if prefix == 'prepare_':
                text = text.replace('scripts/y88_deployment.py', 'scripts/y90_deployment.py')
                text = text.replace('scripts/current_terminal_contract.py', 'scripts/create_y89_terminal_helpers.py')
            generated[prefix + 'y89' + suffix] = text
        assert set(generated) == set(outputs)
        for name, text in generated.items():
            ast.parse(text, feature_version=(3, 8))
            assert not outputs[name].exists()
        for name, text in generated.items():
            outputs[name].write_text(text, encoding='utf-8', newline='\n')
        assert [ref(ROOT / item['path']) for item in originals] == originals
        assert all(ref(path) == value for path, value in before.items())
        assert git('ls-files', '--stage', '-z') == index
        output = ROOT / ('pipeline/checks/y89_terminal_helpers_' + token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, id=CID, spec=spec,
            prepared_at=datetime.now().astimezone().isoformat(), originals=originals,
            evidence=[ref(source), ref(log), ref(location_path), ref(__file__)] +
                [ref(path) for path in outputs.values()],
            source_specialization_only=True, original_helpers_unchanged=True,
            checkpoint_loaded=False, archived=False, queue_changed=False, remote_mutations=False))
        candidate['evidence'].append(ref(output))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(helpers=ref(output), original_checkpoint_directory=spec['checkpoint_directory'],
                              checkpoint_loaded=False, archived=False)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    create(parser.parse_args().token)
