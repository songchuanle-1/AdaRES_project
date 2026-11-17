"""Specialize immutable server-1 checks for the exact live Y92 run."""
import argparse
import ast
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y92_deployment import Y92, git


CID = "AdaRES-Y92"
ADMISSION = "56c0a076cc22dd82200f934781dd50b7a5972ab2adb3c383acb42cb6cb0822dc"
FREEZE = "b866bf8a784b9b44ff63cb1dd7de74fd6288b0b84a1f2280a45c9eaa85a76f95"
EXTRA_TESTS = '''
class CurrentBinding(unittest.TestCase):
    def test_current_original_task_and_actual_declared_location(self):
        from y92_terminal_contract import SPEC, Y92, build_contract
        server, queue, task, work, candidate = archive.current()
        self.assertEqual(build_contract(task, server)['allowed_markers'], [Y92])
        self.assertEqual(task['pid'], 1305468)
        self.assertEqual(candidate['stage'], 'training')
        self.assertEqual(SPEC['log'], task['log'])
        self.assertTrue(SPEC['checkpoint_directory'])
        self.assertEqual(SPEC['admission_sha256'],
            '56c0a076cc22dd82200f934781dd50b7a5972ab2adb3c383acb42cb6cb0822dc')
        self.assertEqual(SPEC['freeze_sha256'],
            'b866bf8a784b9b44ff63cb1dd7de74fd6288b0b84a1f2280a45c9eaa85a76f95')

    def test_actual_GPU1_pause_and_goal_unchanged(self):
        from prepare_y60_stage_heads import ROOT, ref, monitor
        servers = monitor.load_servers(str(ROOT))
        self.assertEqual(servers['server-1']['allowed_gpus'], [0])
        self.assertEqual(servers['server-2']['allowed_gpus'], [0])
        self.assertEqual(servers['server-6']['allowed_gpus'], [])
        self.assertEqual(ref('pipeline/goal.json')['sha256'],
            '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a')
        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'],
            '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')

    def test_old_marker_and_wrong_candidate_refused(self):
        from y92_terminal_contract import Y92, validate_marker
        for marker, cid in [
            ('59baf94e9d3e406afa13b2ddc96968cdcc976cae', 'AdaRES-Y92'),
            (Y92, 'AdaRES-Y91'), ('a' * 40, 'AdaRES-Y92')]:
            with self.subTest(marker=marker, cid=cid), self.assertRaises(AssertionError):
                validate_marker(marker, cid)

    def test_actual_runtime_mapping_and_owner_cannot_change(self):
        from y92_terminal_contract import build_contract
        server, queue, task, work, candidate = archive.current()
        for key, value in [('pid', 2442962), ('server', 'server-2'), ('gpu', '1')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                build_contract(dict(task, **{key: value}), server)
        for key, value in [('python', '/usr/bin/python'), ('expected_hostname', 'foreign'),
                           ('allowed_gpus', [0, 1]), ('runtime_path_mappings', []),
                           ('runtime_library_path', '/wrong/cuda')]:
            with self.subTest(key=key), self.assertRaises((AssertionError, ValueError)):
                build_contract(task, dict(server, **{key: value}))

    def test_actual_native_and_full_parent_protocol_retained(self):
        import copy
        from prepare_y60_stage_heads import ROOT, tick
        from advance_y92 import prepared
        from y92_contract import validate_result
        import closure
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate, _, _, frozen = prepared(work)
        raw = tick.load(closure.artifact(ROOT, candidate['native_smoke_result']))
        validate_result(raw, candidate, frozen)
        current = tick.load(closure.artifact(ROOT, candidate['post_training_freeze']))
        parent_candidate = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y90')
        parent = tick.load(closure.artifact(ROOT, parent_candidate['post_training_freeze']))
        restored = copy.deepcopy(current['evaluation'])
        restored['methods'][0] = 'AdaRES-Y90'
        self.assertEqual(restored, parent['evaluation'])
        self.assertEqual(current['training'], parent['training'])
'''
INJECT = '''
parsed_tests = ast.parse(generated['test_y92_terminal_handoff.py'])
assert sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
           for node in ast.walk(parsed_tests)) == 25
parsed_tests.body[-1:-1] = ast.parse(EXTRA_TESTS).body
generated['test_y92_terminal_handoff.py'] = ast.unparse(ast.fix_missing_locations(parsed_tests)) + '\\n'
class TestCount(ast.NodeTransformer):
    def visit_Constant(self, node):
        if type(node.value) is int and node.value == 25:
            node.value = 30
        elif isinstance(node.value, bytes):
            node.value = node.value.replace(b'Ran 25 tests', b'Ran 30 tests')
        return node
for count_file in ('finalize_y92_fixed30.py', 'prepare_y92_terminal_handoff.py'):
    parsed_count = TestCount().visit(ast.parse(generated[count_file]))
    generated[count_file] = ast.unparse(ast.fix_missing_locations(parsed_count)) + '\\n'
originals.extend([ref('scripts/create_y90_terminal_helpers.py'),
                  ref('pipeline/checks/y92_terminal_creator_' + token + '.py')])
'''


class Identity(ast.NodeTransformer):
    @staticmethod
    def text(value):
        return value.replace("Y90", "Y92").replace("y90", "y92")

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
    assert git("rev-parse", "HEAD").decode().strip() == Y92
    queue = monitor.load_queue(str(ROOT))
    task = next(row for row in queue["gpu_queues"]["server-1"] if row["id"] == CID)
    assert task["status"] == "running" and task["launch_verified"] is True
    assert task["pid"] == 1305468 and task["code_commit"] == Y92
    assert task["config"] == "configs_song/model_paper5/adares_y92_mask_feature_quadratic_res.yaml"
    spec = {key: task[key] for key in ("id", "server", "pid", "code_commit", "config", "log")}
    spec.update(package="adares_y92_mask_feature_quadratic",
                admission_sha256=ADMISSION, freeze_sha256=FREEZE)
    source = ROOT / "scripts/create_y90_terminal_helpers.py"
    source_ref = ref(source)
    tree = Identity().visit(ast.parse(source.read_text(encoding="utf-8")))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "SPEC" for target in node.targets)]
    assert len(assignments) == 1
    assignments[0].value = ast.parse(repr(spec), mode="eval").body
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "create")
    try_block = next(node for node in function.body if isinstance(node, ast.Try))
    positions = [index for index, node in enumerate(try_block.body)
                 if isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                 and isinstance(node.iter.func, ast.Attribute) and node.iter.func.attr == "items"
                 and isinstance(node.iter.func.value, ast.Name) and node.iter.func.value.id == "generated"
                 and any(isinstance(child, ast.Assert) for child in node.body)]
    assert len(positions) == 1
    position = positions[0]
    try_block.body[position:position] = ast.parse(INJECT).body
    ast.fix_missing_locations(tree)
    body = "EXTRA_TESTS=" + repr(EXTRA_TESTS) + "\n" + ast.unparse(tree) + "\n"
    ast.parse(body, feature_version=(3, 8))
    program = ROOT / ("pipeline/checks/y92_terminal_creator_" + token + ".py")
    assert not program.exists()
    program.write_text(body, encoding="utf-8", newline="\n")
    namespace = {"__name__": "y92_exact_terminal_creator", "__file__": __file__}
    exec(compile(body, str(program), "exec"), namespace)
    namespace["create"](token)
    assert ref(source) == source_ref
    print(json.dumps(dict(creator=ref(program), immutable_template=source_ref,
                          lifecycle_and_binding_guards_prepared=30,
                          checkpoint_loaded=False, archived=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    create(parser.parse_args().token)
