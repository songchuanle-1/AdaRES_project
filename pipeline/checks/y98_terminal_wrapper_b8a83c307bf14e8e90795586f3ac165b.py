"""Prepare exact Y98 lifecycle helpers from preserved server-2 templates."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import subprocess
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finalize_y61_fixed30 import DISCOVER
from y84_terminal_contract import IDENTITY
from create_y89_terminal_helpers import LIVE_IDENTITY
from y98_deployment import Y98, contract, git, paths, adapt
import closure
CID = 'AdaRES-Y98'
SCRIPTS = ROOT / 'scripts'
SPEC = dict(id=CID, server='server-2', pid=25962, code_commit=Y98, package='adares_y98_appearance_mask_skip', config='configs_song/model_paper5/adares_y98_appearance_mask_skip_res.yaml', log='/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y98_appearance_mask_skip_res_20261110_210741.log', admission_sha256='dc1a8a6dd3afde694d255cceca30b1b16009e8249390aa706632e2a567136a62', freeze_sha256='bcd19eedca805c8491568b3f81cf9cefa8dc4b0def8f6a09dd87c58f5e29d51c')
CONTRACT = "\nCID = SPEC['id']\nCHECKPOINT_DIRECTORY = SPEC['checkpoint_directory']\nCONFIG = SPEC['config']\n\n\ndef validate_marker(marker, candidate):\n    assert candidate == CID and marker == Y98\n\n\ndef build_contract(task, server):\n    assert task['id'] == CID and task['status'] == 'running' and task['launch_verified'] is True\n    assert str(task['gpu']) == '0'\n    assert all(task[key] == SPEC[key] for key in ('server', 'pid', 'code_commit', 'config', 'log'))\n    closure.validate_adares_server(ROOT, server)\n    assert server['allowed_gpus'] == [0] and server['max_tasks_per_gpu'] == 1\n    for key, value in dict(host='192.168.1.163', expected_hostname='dl01-MS-7C60',\n            python='/home/songcl/.conda/envs/adares_project/bin/python',\n            remote_code_root='/bak_home/songcl/Project/AdaRES_project/AdaRES',\n            work_dir='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints').items():\n        assert server[key] == value\n    assert not server.get('runtime_path_mappings')\n    assert ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'] == (\n        '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')\n    location = json.loads(closure.artifact(ROOT, SPEC['location_evidence']).read_text(encoding='utf-8'))\n    assert location['checkpoint_loaded'] is False and location['remote_mutations'] is False\n    assert location['log'] == SPEC['log'] and location['checkpoint_directory'] == CHECKPOINT_DIRECTORY\n    expected = PurePosixPath(server['work_dir']) / PurePosixPath(CONFIG).stem / CHECKPOINT_DIRECTORY / 'last.pth'\n    assert PurePosixPath(location['declared_last_path']) == expected\n    assert git('rev-parse', 'HEAD').decode().strip() == Y98\n    deployments = contract(Y98, Y98, CID)\n    assert set(deployments) == {Y98}\n    for deployment in deployments.values():\n        deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}\n    return dict(allowed_markers=list(deployments), deployments=deployments)\n"
EXTRA_TESTS = "\n\nclass CurrentBinding(unittest.TestCase):\n    def test_actual_current_task_source_YAML_and_frozen_inputs(self):\n        from y98_terminal_contract import build_contract, Y98\n        server, queue, task, work, candidate = archive.current()\n        self.assertEqual(build_contract(task, server)['allowed_markers'], [Y98])\n        self.assertEqual(task['pid'], 25962)\n        self.assertEqual(candidate['stage'], 'training')\n\n    def test_actual_GPU1_pause_and_goal_bytes_retained(self):\n        from prepare_y60_stage_heads import ROOT, ref, monitor\n        servers = monitor.load_servers(str(ROOT))\n        self.assertEqual(servers['server-1']['allowed_gpus'], [0])\n        self.assertEqual(servers['server-2']['allowed_gpus'], [0])\n        self.assertEqual(servers['server-6']['allowed_gpus'], [])\n        self.assertEqual(ref('pipeline/goal.json')['sha256'],\n            '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a')\n        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'],\n            '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')\n\n    def test_old_marker_and_wrong_candidate_refused(self):\n        from y98_terminal_contract import validate_marker, Y98\n        for marker, cid in [\n            ('2d2cccb4f6f6ef9daf960b0ac255a3a3ee5e9f71', 'AdaRES-Y98'),\n            (Y98, 'AdaRES-Y90'), ('a' * 40, 'AdaRES-Y98')]:\n            with self.subTest(marker=marker, cid=cid), self.assertRaises(AssertionError):\n                validate_marker(marker, cid)\n\n    def test_actual_terminal_contract_rejects_wrong_runtime_and_owner(self):\n        from y98_terminal_contract import build_contract\n        server, queue, task, work, candidate = archive.current()\n        for key, value in [('pid', 60578), ('server', 'server-1'), ('gpu', '1')]:\n            with self.subTest(key=key), self.assertRaises(AssertionError):\n                build_contract(dict(task, **{key: value}), server)\n        for key, value in [('python', '/usr/bin/python'), ('expected_hostname', 'foreign'),\n                           ('allowed_gpus', [0, 1])]:\n            with self.subTest(key=key), self.assertRaises((AssertionError, ValueError)):\n                build_contract(task, dict(server, **{key: value}))\n"

class Specialize(ast.NodeTransformer):

    def __init__(self, directory):
        self.replacements = (('adares_y89_stage_evidence_norm', SPEC['package']), ('20261106_122357', directory), ('5f1c4bde4b682fabd84fdc84a4341ba2c845fe71', Y98), ('0ebbe5d0997ad2a3ac8e2d89ba11cbb9e5d47bd0b5e3710de9aa24a06d0220ec', SPEC['admission_sha256']), ('4ea48fe7d44504047205956a929b32dc18a094319f975707ca5f806c54d2347c', SPEC['freeze_sha256']), ('scripts/y90_deployment.py', 'scripts/y98_deployment.py'), ('Y89', 'Y98'), ('y89', 'y98'))

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for old, new in self.replacements:
                node.value = node.value.replace(old, new)
        elif type(node.value) is int:
            node.value = {60578: 25962, 25: 29}.get(node.value, node.value)
        elif isinstance(node.value, bytes):
            node.value = node.value.replace(b'Ran 25 tests', b'Ran 29 tests')
        return node

    def visit_ImportFrom(self, node):
        node.module = node.module.replace('y89', 'y98')
        return node

    def visit_Import(self, node):
        for item in node.names:
            item.name = item.name.replace('y89', 'y98')
        return node

    def visit_FunctionDef(self, node):
        if node.name == 'test_exact_registered_production_markers_only':
            return ast.parse("\ndef test_exact_registered_production_markers_only(self):\n    from y98_terminal_contract import Y98, validate_marker\n    validate_marker(Y98, 'AdaRES-Y98')\n    with self.assertRaises(AssertionError):\n        validate_marker('a' * 40, 'AdaRES-Y98')\n").body[0]
        node = self.generic_visit(node)
        if node.name == 'test_additive_edges_reject_foreign_or_modified_paths':
            for item in ast.walk(node):
                if isinstance(item, ast.Constant) and item.value == 'AdaRES-Y90':
                    item.value = CID
        return node

def create(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y98
        before = {path: ref(path) for path in ('pipeline/queue.json', 'pipeline/running.json', 'pipeline/goal.json', 'SERVERS.yaml')}
        index = git('ls-files', '--stage', '-z')
        queue = monitor.load_queue(str(ROOT))
        task = next((row for row in queue['gpu_queues']['server-2'] if row['id'] == CID))
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next((row for row in work['candidates'] if row['id'] == CID))
        assert candidate['stage'] == 'training' and (not candidate.get('terminal_handoff_plan'))
        assert task['status'] == 'running' and task['launch_verified'] is True and (str(task['gpu']) == '0')
        assert all((task[key] == SPEC[key] for key in ('server', 'pid', 'code_commit', 'config', 'log')))
        assert monitor.load_running(str(ROOT))['server-2']['0'] == CID
        server = monitor.load_servers(str(ROOT))['server-2']
        closure.validate_adares_server(ROOT, server)
        closure.admission(ROOT, task)
        assert ref('pipeline/admissions/' + CID + '.json')['sha256'] == SPEC['admission_sha256']
        assert candidate['post_training_freeze']['sha256'] == SPEC['freeze_sha256']
        closure.artifact(ROOT, candidate['post_training_freeze'])
        outputs = {name: SCRIPTS / name for name in ('finalize_y98_fixed30.py', 'test_y98_terminal_handoff.py', 'prepare_y98_terminal_handoff.py', 'y98_terminal_contract.py')}
        assert not any((path.exists() for path in outputs.values()))
        originals = [ref('scripts/' + name) for name in ('finalize_y89_fixed30.py', 'test_y89_terminal_handoff.py', 'prepare_y89_terminal_handoff.py', 'y89_terminal_contract.py')]
        deployments = contract(Y98, Y98, CID)
        for deployment in deployments.values():
            deployment['configs'] = {path: adapt(value, server) for path, value in deployment['configs'].items()}
        config = __import__('yaml').safe_load((ROOT / 'AdaRES' / SPEC['config']).read_bytes())
        payload = dict(id=CID, pid=SPEC['pid'], root=server['remote_code_root'], work=server['work_dir'], config=SPEC['config'], log=SPEC['log'], basename=Path(SPEC['config']).stem, code_commit=Y98, expected_config=adapt(config, server), source_files={path[len('AdaRES/'):]: ref(path)['sha256'] for path in paths(CID)[:-1]}, deployments=deployments, allowed_markers=list(deployments))
        body = DISCOVER.replace('print(json.dumps(dict(complete=True', IDENTITY + LIVE_IDENTITY + 'print(json.dumps(dict(complete=True')
        program = 'P=' + repr(payload) + '\n' + body
        ast.parse(program, feature_version=(3, 8))
        base = ROOT / ('pipeline/checks/y98_terminal_original_location_' + token)
        location_path, source, log = (base.with_suffix('.json'), base.with_suffix('.py'), base.with_suffix('.log'))
        assert not any((path.exists() for path in (location_path, source, log)))
        source.write_text(program, encoding='utf-8', newline='\n')
        response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', server['user'] + '@' + server['host'], 'PYTHONDONTWRITEBYTECODE=1 ' + server['python'] + ' -B -'], input=program.encode(), capture_output=True, timeout=45)
        log.write_bytes(response.stdout + response.stderr)
        assert response.returncode == 0, 'Preserve failed read-only discovery; never restart'
        value = json.loads(response.stdout)
        assert value['complete'] is True and value['checkpoint_loaded'] is False and (value['remote_mutations'] is False)
        assert value['log'] == SPEC['log']
        tick.save(location_path, value)
        spec = dict(SPEC, checkpoint_directory=value['checkpoint_directory'], location_evidence=ref(location_path))
        generated = {'y98_terminal_contract.py': '"""Exact original Y98 terminal contract; future HEADs require a new tested binding."""\nimport json\nfrom pathlib import PurePosixPath\nfrom prepare_y60_stage_heads import ROOT, ref\nfrom y84_terminal_contract import IDENTITY\nfrom y98_deployment import Y98, contract, git, paths, adapt, validate_delta\nimport closure\n\nSPEC = ' + repr(spec) + '\n' + CONTRACT}
        for prefix, suffix in (('finalize_', '_fixed30.py'), ('test_', '_terminal_handoff.py'), ('prepare_', '_terminal_handoff.py')):
            original = SCRIPTS / (prefix + 'y89' + suffix)
            parsed = Specialize(spec['checkpoint_directory']).visit(ast.parse(original.read_text(encoding='utf-8')))
            if prefix == 'test_':
                parsed.body[-1:-1] = ast.parse(EXTRA_TESTS).body
            text = ast.unparse(ast.fix_missing_locations(parsed)) + '\n'
            generated[prefix + 'y98' + suffix] = text
        assert set(generated) == set(outputs)
        for name, text in generated.items():
            ast.parse(text, feature_version=(3, 8))
            assert not outputs[name].exists()
        for name, text in generated.items():
            outputs[name].write_text(text, encoding='utf-8', newline='\n')
        assert [ref(ROOT / item['path']) for item in originals] == originals
        assert all((ref(path) == value for path, value in before.items()))
        assert git('ls-files', '--stage', '-z') == index
        output = ROOT / ('pipeline/checks/y98_terminal_helpers_' + token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, id=CID, spec=spec, prepared_at=datetime.now().astimezone().isoformat(), originals=originals, evidence=[ref(source), ref(log), ref(location_path), ref(__file__)] + [ref(path) for path in outputs.values()], source_specialization_only=True, original_helpers_unchanged=True, checkpoint_loaded=False, archived=False, queue_changed=False, remote_mutations=False))
        candidate['evidence'].append(ref(output))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(helpers=ref(output), original_checkpoint_directory=spec['checkpoint_directory'], checkpoint_loaded=False, archived=False)))
    finally:
        writer.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    create(parser.parse_args().token)
