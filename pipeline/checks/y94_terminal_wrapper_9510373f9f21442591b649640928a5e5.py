"""Specialize immutable server-1 checks for the exact live Y94 run."""
import argparse
import ast
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y94_deployment import Y94, git
CID = 'AdaRES-Y94'
ADMISSION = 'b9deb51efa3c274283f8a23ae678d42a4f0c11a29e758d19ae696e80df13cb84'
FREEZE = '7e40371552d170a1d764f9dab160a8dcb12b0b6fe512e5c978313cb7b642241b'
EXTRA_TESTS = "\nclass CurrentBinding(unittest.TestCase):\n    def test_current_original_task_and_actual_declared_location(self):\n        from y94_terminal_contract import SPEC, Y94, build_contract\n        server, queue, task, work, candidate = archive.current()\n        self.assertEqual(build_contract(task, server)['allowed_markers'], [Y94])\n        self.assertEqual(task['pid'], 218513)\n        self.assertEqual(candidate['stage'], 'training')\n        self.assertEqual(SPEC['log'], task['log'])\n        self.assertTrue(SPEC['checkpoint_directory'])\n        self.assertEqual(SPEC['admission_sha256'],\n            'b9deb51efa3c274283f8a23ae678d42a4f0c11a29e758d19ae696e80df13cb84')\n        self.assertEqual(SPEC['freeze_sha256'],\n            '7e40371552d170a1d764f9dab160a8dcb12b0b6fe512e5c978313cb7b642241b')\n\n    def test_actual_GPU1_pause_and_goal_unchanged(self):\n        from prepare_y60_stage_heads import ROOT, ref, monitor\n        servers = monitor.load_servers(str(ROOT))\n        self.assertEqual(servers['server-1']['allowed_gpus'], [0])\n        self.assertEqual(servers['server-2']['allowed_gpus'], [0])\n        self.assertEqual(servers['server-6']['allowed_gpus'], [])\n        self.assertEqual(ref('pipeline/goal.json')['sha256'],\n            '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a')\n        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'],\n            '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')\n\n    def test_old_marker_and_wrong_candidate_refused(self):\n        from y94_terminal_contract import Y94, validate_marker\n        for marker, cid in [\n            ('59baf94e9d3e406afa13b2ddc96968cdcc976cae', 'AdaRES-Y94'),\n            (Y94, 'AdaRES-Y91'), ('a' * 40, 'AdaRES-Y94')]:\n            with self.subTest(marker=marker, cid=cid), self.assertRaises(AssertionError):\n                validate_marker(marker, cid)\n\n    def test_actual_runtime_mapping_and_owner_cannot_change(self):\n        from y94_terminal_contract import build_contract\n        server, queue, task, work, candidate = archive.current()\n        for key, value in [('pid', 1305468), ('server', 'server-2'), ('gpu', '1')]:\n            with self.subTest(key=key), self.assertRaises(AssertionError):\n                build_contract(dict(task, **{key: value}), server)\n        for key, value in [('python', '/usr/bin/python'), ('expected_hostname', 'foreign'),\n                           ('allowed_gpus', [0, 1]), ('runtime_path_mappings', []),\n                           ('runtime_library_path', '/wrong/cuda')]:\n            with self.subTest(key=key), self.assertRaises((AssertionError, ValueError)):\n                build_contract(task, dict(server, **{key: value}))\n\n    def test_actual_native_and_full_parent_protocol_retained(self):\n        import copy\n        from prepare_y60_stage_heads import ROOT, tick\n        from advance_y94 import prepared\n        from y94_contract import validate_result\n        import closure\n        work = tick.load(ROOT / 'pipeline/research_work.json')\n        candidate, _, _, frozen = prepared(work)\n        raw = tick.load(closure.artifact(ROOT, candidate['native_smoke_result']))\n        validate_result(raw, candidate, frozen)\n        current = tick.load(closure.artifact(ROOT, candidate['post_training_freeze']))\n        parent_candidate = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y92')\n        parent = tick.load(closure.artifact(ROOT, parent_candidate['post_training_freeze']))\n        restored = copy.deepcopy(current['evaluation'])\n        restored['methods'][0] = 'AdaRES-Y92'\n        self.assertEqual(restored, parent['evaluation'])\n        self.assertEqual(current['training'], parent['training'])\n"
INJECT = "\nparsed_tests = ast.parse(generated['test_y94_terminal_handoff.py'])\nassert sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_')\n           for node in ast.walk(parsed_tests)) == 25\nparsed_tests.body[-1:-1] = ast.parse(EXTRA_TESTS).body\ngenerated['test_y94_terminal_handoff.py'] = ast.unparse(ast.fix_missing_locations(parsed_tests)) + '\\n'\nclass TestCount(ast.NodeTransformer):\n    def visit_Constant(self, node):\n        if type(node.value) is int and node.value == 25:\n            node.value = 30\n        elif isinstance(node.value, bytes):\n            node.value = node.value.replace(b'Ran 25 tests', b'Ran 30 tests')\n        return node\nfor count_file in ('finalize_y94_fixed30.py', 'prepare_y94_terminal_handoff.py'):\n    parsed_count = TestCount().visit(ast.parse(generated[count_file]))\n    generated[count_file] = ast.unparse(ast.fix_missing_locations(parsed_count)) + '\\n'\noriginals.extend([ref('scripts/create_y90_terminal_helpers.py'),\n                  ref('pipeline/checks/y94_terminal_creator_' + token + '.py')])\n"

class Identity(ast.NodeTransformer):

    @staticmethod
    def text(value):
        return value.replace('Y90', 'Y94').replace('y90', 'y94')

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.text(node.value)
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

def create(token):
    lease(token, 300)
    assert git('rev-parse', 'HEAD').decode().strip() == Y94
    queue = monitor.load_queue(str(ROOT))
    task = next((row for row in queue['gpu_queues']['server-1'] if row['id'] == CID))
    assert task['status'] == 'running' and task['launch_verified'] is True
    assert task['pid'] == 218513 and task['code_commit'] == Y94
    assert task['config'] == 'configs_song/model_paper5/adares_y94_query_geometry_bias_res.yaml'
    spec = {key: task[key] for key in ('id', 'server', 'pid', 'code_commit', 'config', 'log')}
    spec.update(package='adares_y94_query_geometry_bias', admission_sha256=ADMISSION, freeze_sha256=FREEZE)
    source = ROOT / 'scripts/create_y90_terminal_helpers.py'
    source_ref = ref(source)
    tree = Identity().visit(ast.parse(source.read_text(encoding='utf-8')))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign) and any((isinstance(target, ast.Name) and target.id == 'SPEC' for target in node.targets))]
    assert len(assignments) == 1
    assignments[0].value = ast.parse(repr(spec), mode='eval').body
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'create'))
    try_block = next((node for node in function.body if isinstance(node, ast.Try)))
    positions = [index for index, node in enumerate(try_block.body) if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute) and (node.iter.func.attr == 'items') and isinstance(node.iter.func.value, ast.Name) and (node.iter.func.value.id == 'generated') and any((isinstance(child, ast.Assert) for child in node.body))]
    assert len(positions) == 1
    position = positions[0]
    try_block.body[position:position] = ast.parse(INJECT).body
    ast.fix_missing_locations(tree)
    body = 'EXTRA_TESTS=' + repr(EXTRA_TESTS) + '\n' + ast.unparse(tree) + '\n'
    ast.parse(body, feature_version=(3, 8))
    program = ROOT / ('pipeline/checks/y94_terminal_creator_' + token + '.py')
    assert not program.exists()
    program.write_text(body, encoding='utf-8', newline='\n')
    namespace = {'__name__': 'y94_exact_terminal_creator', '__file__': __file__}
    exec(compile(body, str(program), 'exec'), namespace)
    namespace['create'](token)
    assert ref(source) == source_ref
    print(json.dumps(dict(creator=ref(program), immutable_template=source_ref, lifecycle_and_binding_guards_prepared=30, checkpoint_loaded=False, archived=False)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    create(parser.parse_args().token)
