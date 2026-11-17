EXTRA_TESTS="\nclass CurrentBinding(unittest.TestCase):\n    def test_current_original_task_and_actual_declared_location(self):\n        from y92_terminal_contract import SPEC, Y92, build_contract\n        server, queue, task, work, candidate = archive.current()\n        self.assertEqual(build_contract(task, server)['allowed_markers'], [Y92])\n        self.assertEqual(task['pid'], 1305468)\n        self.assertEqual(candidate['stage'], 'training')\n        self.assertEqual(SPEC['log'], task['log'])\n        self.assertTrue(SPEC['checkpoint_directory'])\n        self.assertEqual(SPEC['admission_sha256'],\n            '56c0a076cc22dd82200f934781dd50b7a5972ab2adb3c383acb42cb6cb0822dc')\n        self.assertEqual(SPEC['freeze_sha256'],\n            'b866bf8a784b9b44ff63cb1dd7de74fd6288b0b84a1f2280a45c9eaa85a76f95')\n\n    def test_actual_GPU1_pause_and_goal_unchanged(self):\n        from prepare_y60_stage_heads import ROOT, ref, monitor\n        servers = monitor.load_servers(str(ROOT))\n        self.assertEqual(servers['server-1']['allowed_gpus'], [0])\n        self.assertEqual(servers['server-2']['allowed_gpus'], [0])\n        self.assertEqual(servers['server-6']['allowed_gpus'], [])\n        self.assertEqual(ref('pipeline/goal.json')['sha256'],\n            '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a')\n        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'],\n            '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')\n\n    def test_old_marker_and_wrong_candidate_refused(self):\n        from y92_terminal_contract import Y92, validate_marker\n        for marker, cid in [\n            ('59baf94e9d3e406afa13b2ddc96968cdcc976cae', 'AdaRES-Y92'),\n            (Y92, 'AdaRES-Y91'), ('a' * 40, 'AdaRES-Y92')]:\n            with self.subTest(marker=marker, cid=cid), self.assertRaises(AssertionError):\n                validate_marker(marker, cid)\n\n    def test_actual_runtime_mapping_and_owner_cannot_change(self):\n        from y92_terminal_contract import build_contract\n        server, queue, task, work, candidate = archive.current()\n        for key, value in [('pid', 2442962), ('server', 'server-2'), ('gpu', '1')]:\n            with self.subTest(key=key), self.assertRaises(AssertionError):\n                build_contract(dict(task, **{key: value}), server)\n        for key, value in [('python', '/usr/bin/python'), ('expected_hostname', 'foreign'),\n                           ('allowed_gpus', [0, 1]), ('runtime_path_mappings', []),\n                           ('runtime_library_path', '/wrong/cuda')]:\n            with self.subTest(key=key), self.assertRaises((AssertionError, ValueError)):\n                build_contract(task, dict(server, **{key: value}))\n\n    def test_actual_native_and_full_parent_protocol_retained(self):\n        import copy\n        from prepare_y60_stage_heads import ROOT, tick\n        from advance_y92 import prepared\n        from y92_contract import validate_result\n        import closure\n        work = tick.load(ROOT / 'pipeline/research_work.json')\n        candidate, _, _, frozen = prepared(work)\n        raw = tick.load(closure.artifact(ROOT, candidate['native_smoke_result']))\n        validate_result(raw, candidate, frozen)\n        current = tick.load(closure.artifact(ROOT, candidate['post_training_freeze']))\n        parent_candidate = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y90')\n        parent = tick.load(closure.artifact(ROOT, parent_candidate['post_training_freeze']))\n        restored = copy.deepcopy(current['evaluation'])\n        restored['methods'][0] = 'AdaRES-Y90'\n        self.assertEqual(restored, parent['evaluation'])\n        self.assertEqual(current['training'], parent['training'])\n"
"""Specialize existing server-1 lifecycle checks for the exact live Y92 run."""
import argparse
import ast
from datetime import datetime
from pathlib import Path, PurePosixPath
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finalize_y61_fixed30 import DISCOVER
from finalize_y88_fixed30 import remote
from y84_terminal_contract import IDENTITY
from y92_deployment import Y92, adapt, contract, git, paths
import closure
import yaml
CID = 'AdaRES-Y92'
SCRIPTS = ROOT / 'scripts'
SPEC = {'id': 'AdaRES-Y92', 'server': 'server-1', 'pid': 1305468, 'code_commit': '27d6f67e7121a5168bbfeca9db745377e0bd49e5', 'config': 'configs_song/model_paper5/adares_y92_mask_feature_quadratic_res.yaml', 'log': '/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y92_mask_feature_quadratic_res_20261108_090155.log', 'package': 'adares_y92_mask_feature_quadratic', 'admission_sha256': '56c0a076cc22dd82200f934781dd50b7a5972ab2adb3c383acb42cb6cb0822dc', 'freeze_sha256': 'b866bf8a784b9b44ff63cb1dd7de74fd6288b0b84a1f2280a45c9eaa85a76f95'}
MAPPINGS = [dict(source='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data', target='/data1/songcl/Project/AdaRES_project/data'), dict(source='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', target='/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints')]
LIVE_IDENTITY = "\nprocess=pathlib.Path('/proc')/str(P['pid'])\nassert process.is_dir() and (process/'cwd').resolve()==pathlib.Path(P['root'])\nargs=(process/'cmdline').read_bytes().split(b'\\0')\nassert b'tools/train.py' in args and P['config'].encode() in args\n"
CONTRACT = '\nCID = SPEC["id"]\nCHECKPOINT_DIRECTORY = SPEC["checkpoint_directory"]\nCONFIG = SPEC["config"]\n\n\ndef validate_marker(marker, candidate):\n    assert candidate == CID and marker == Y92\n\n\ndef validate_server(server):\n    closure.validate_adares_server(ROOT, server)\n    assert server["allowed_gpus"] == [0] and server["max_tasks_per_gpu"] == 1\n    for key, value in dict(\n        host="192.168.1.175", expected_hostname="dl02-MS-7B73",\n        python="/data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python",\n        runtime_library_path="/data1/songcl/Project/AdaRES_project/runtime/cuda113lib64",\n        remote_code_root="/data1/songcl/Project/AdaRES_project/AdaRES",\n        work_dir="/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints",\n    ).items():\n        assert server[key] == value\n    assert server["runtime_path_mappings"] == MAPPINGS\n\n\ndef build_contract(task, server):\n    assert task["id"] == CID and task["status"] == "running"\n    assert task["launch_verified"] is True and str(task["gpu"]) == "0"\n    assert all(task[key] == SPEC[key] for key in\n               ("server", "pid", "code_commit", "config", "log"))\n    validate_server(server)\n    location = json.loads(closure.artifact(ROOT, SPEC["location_evidence"]).read_text(encoding="utf-8"))\n    assert location["complete"] is True\n    assert location["checkpoint_loaded"] is False and location["remote_mutations"] is False\n    assert location["log"] == SPEC["log"] and location["checkpoint_directory"] == CHECKPOINT_DIRECTORY\n    expected = PurePosixPath(server["work_dir"]) / PurePosixPath(CONFIG).stem / CHECKPOINT_DIRECTORY / "last.pth"\n    assert PurePosixPath(location["declared_last_path"]) == expected\n    assert git("rev-parse", "HEAD").decode().strip() == Y92\n    deployments = contract(Y92, Y92, CID)\n    assert set(deployments) == {Y92}\n    for deployment in deployments.values():\n        deployment["configs"] = {path: adapt(value, server) for path, value in deployment["configs"].items()}\n    return dict(allowed_markers=list(deployments), deployments=deployments)\n'

class Specialize(ast.NodeTransformer):

    def __init__(self, directory):
        self.replacements = [('adares_y88_mask_query_quadratic', SPEC['package']), ('20261106_053553', directory), ('cb4a5813755d38b3d1593bb803726aabd7d0a79b', Y92), ('39f9a71debe48314d479b42b604565cf557ccbe40e5582e949a66959587b3572', SPEC['admission_sha256']), ('93bf51b65a580cbdd5a7ade69fae370d8d0e6e6db5ab665645892c775d84aa81', SPEC['freeze_sha256']), ('scripts/current_terminal_contract.py', 'scripts/create_y92_terminal_helpers.py'), ('Y88', 'Y92'), ('y88', 'y92')]

    def text(self, value):
        for old, new in self.replacements:
            value = value.replace(old, new)
        return value

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.text(node.value)
        elif type(node.value) is int and node.value == 3527278:
            node.value = SPEC['pid']
        return node

    def visit_ImportFrom(self, node):
        node.module = self.text(node.module)
        return self.generic_visit(node)

    def visit_alias(self, node):
        node.name = self.text(node.name)
        return node

    def visit_Name(self, node):
        node.id = self.text(node.id)
        return node

    def visit_keyword(self, node):
        if node.arg:
            node.arg = self.text(node.arg)
        return self.generic_visit(node)

def create(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y92
        before = {path: ref(path) for path in ('pipeline/queue.json', 'pipeline/running.json', 'pipeline/goal.json', 'SERVERS.yaml')}
        index = git('ls-files', '--stage', '-z')
        work = tick.load(ROOT / 'pipeline/research_work.json')
        original_candidates = {row['id']: row for row in work['candidates'] if row['id'] != CID}
        candidate = next((row for row in work['candidates'] if row['id'] == CID))
        assert candidate['stage'] == 'training' and (not candidate.get('terminal_handoff_plan'))
        assert not candidate.get('terminal_evidence')
        queue = monitor.load_queue(str(ROOT))
        task = next((row for row in queue['gpu_queues']['server-1'] if row['id'] == CID))
        assert task['status'] == 'running' and task['launch_verified'] is True
        assert str(task['gpu']) == '0'
        assert all((task[key] == SPEC[key] for key in ('id', 'server', 'pid', 'code_commit', 'config', 'log')))
        assert monitor.load_running(str(ROOT))['server-1']['0'] == CID
        server = monitor.load_servers(str(ROOT))['server-1']
        closure.validate_adares_server(ROOT, server)
        assert server['runtime_path_mappings'] == MAPPINGS
        assert server['python'] == '/data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python'
        assert server['runtime_library_path'] == '/data1/songcl/Project/AdaRES_project/runtime/cuda113lib64'
        closure.admission(ROOT, task)
        assert ref('pipeline/admissions/' + CID + '.json')['sha256'] == SPEC['admission_sha256']
        assert candidate['post_training_freeze']['sha256'] == SPEC['freeze_sha256']
        closure.artifact(ROOT, candidate['post_training_freeze'])
        names = ('finalize_y92_fixed30.py', 'test_y92_terminal_handoff.py', 'prepare_y92_terminal_handoff.py', 'y92_terminal_contract.py')
        outputs = {name: SCRIPTS / name for name in names}
        assert not any((path.exists() for path in outputs.values()))
        originals = [ref('scripts/' + name.replace('y92', 'y88')) for name in names]
        originals.extend((ref('scripts/' + name) for name in ('finalize_y89_fixed30.py', 'prepare_y89_terminal_handoff.py', 'test_y89_terminal_handoff.py', 'y89_terminal_contract.py')))
        config = yaml.safe_load((ROOT / 'AdaRES' / SPEC['config']).read_bytes())
        deployments = contract(Y92, Y92, CID)
        payload = dict(id=CID, pid=SPEC['pid'], root=server['remote_code_root'], work=server['work_dir'], config=SPEC['config'], log=SPEC['log'], basename=Path(SPEC['config']).stem, code_commit=Y92, expected_config=adapt(config, server), source_files={path[len('AdaRES/'):]: ref(path)['sha256'] for path in paths(CID)[:-1]}, deployments=deployments, allowed_markers=list(deployments))
        body = DISCOVER.replace('dl01-MS-7C60', 'dl02-MS-7B73').replace('print(json.dumps(dict(complete=True', IDENTITY + LIVE_IDENTITY + 'print(json.dumps(dict(complete=True')
        program = 'P=' + repr(payload) + '\n' + body
        ast.parse(program, feature_version=(3, 8))
        base = ROOT / ('pipeline/checks/y92_terminal_original_location_' + token)
        source, location_path = (base.with_suffix('.py'), base.with_suffix('.json'))
        assert not source.exists() and (not location_path.exists())
        source.write_text(program, encoding='utf-8', newline='\n')
        location = remote(server, program, location_path)
        assert location['complete'] is True
        assert location['checkpoint_loaded'] is False and location['remote_mutations'] is False
        assert location['log'] == SPEC['log']
        directory = location['checkpoint_directory']
        expected = PurePosixPath(server['work_dir']) / payload['basename'] / directory / 'last.pth'
        assert PurePosixPath(location['declared_last_path']) == expected
        tick.save(location_path, location)
        spec = dict(SPEC, checkpoint_directory=directory, location_evidence=ref(location_path))
        generated = {'y92_terminal_contract.py': '"""Exact original Y92 run; future source advances require a new tested binding."""\nimport json\nfrom pathlib import PurePosixPath\nfrom prepare_y60_stage_heads import ROOT\nfrom y84_terminal_contract import IDENTITY\nfrom y92_deployment import Y92, adapt, contract, git, paths, validate_delta\nimport closure\n\nSPEC = ' + repr(spec) + '\nMAPPINGS = ' + repr(MAPPINGS) + '\n' + CONTRACT}
        for name in names[:-1]:
            template = (SCRIPTS / name.replace('y92', 'y88')).read_text(encoding='utf-8')
            tree = Specialize(directory).visit(ast.parse(template))
            ast.fix_missing_locations(tree)
            generated[name] = ast.unparse(tree) + '\n'
        parsed_tests = ast.parse(generated['test_y92_terminal_handoff.py'])
        assert sum((isinstance(node, ast.FunctionDef) and node.name.startswith('test_') for node in ast.walk(parsed_tests))) == 25
        parsed_tests.body[-1:-1] = ast.parse(EXTRA_TESTS).body
        generated['test_y92_terminal_handoff.py'] = ast.unparse(ast.fix_missing_locations(parsed_tests)) + '\n'

        class TestCount(ast.NodeTransformer):

            def visit_Constant(self, node):
                if type(node.value) is int and node.value == 25:
                    node.value = 30
                elif isinstance(node.value, bytes):
                    node.value = node.value.replace(b'Ran 25 tests', b'Ran 30 tests')
                return node
        for count_file in ('finalize_y92_fixed30.py', 'prepare_y92_terminal_handoff.py'):
            parsed_count = TestCount().visit(ast.parse(generated[count_file]))
            generated[count_file] = ast.unparse(ast.fix_missing_locations(parsed_count)) + '\n'
        originals.extend([ref('scripts/create_y90_terminal_helpers.py'), ref('pipeline/checks/y92_terminal_creator_' + token + '.py')])
        for name, text in generated.items():
            ast.parse(text, feature_version=(3, 8))
            assert 'adares_y88_mask_query_quadratic' not in text
            assert '3527278' not in text and '20261106_053553' not in text
            assert not outputs[name].exists()
        for name, text in generated.items():
            outputs[name].write_text(text, encoding='utf-8', newline='\n')
        assert [ref(ROOT / item['path']) for item in originals] == originals
        assert all((ref(path) == value for path, value in before.items()))
        assert git('ls-files', '--stage', '-z') == index
        assert {row['id']: row for row in work['candidates'] if row['id'] != CID} == original_candidates
        output = ROOT / ('pipeline/checks/y92_terminal_helpers_' + token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, id=CID, spec=spec, prepared_at=datetime.now().astimezone().isoformat(), originals=originals, evidence=[ref(source), ref(location_path), ref(location_path.with_suffix('.log')), ref(__file__)] + [ref(path) for path in outputs.values()], source_specialization_only=True, original_helpers_unchanged=True, checkpoint_loaded=False, archived=False, queue_changed=False, remote_mutations=False, main_index_unchanged=True, other_candidates_unchanged=True))
        candidate['evidence'].append(ref(output))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(__import__('json').dumps(dict(helpers=ref(output), original_checkpoint_directory=directory, checkpoint_loaded=False, archived=False)))
    finally:
        writer.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    create(parser.parse_args().token)
