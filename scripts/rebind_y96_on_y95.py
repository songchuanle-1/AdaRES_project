"""Rebind the unchanged tested Y96 files to actual-native Y95 and current owners."""
import argparse
import ast
import json
import re

from prepare_y60_stage_heads import ROOT, lease, ref
from y95_deployment import Y95, git


OLD_PARENT = "daa93b4375f144f0e79da92f85fe1fe27e877e18"
OLD_COMMIT = "994db457a3b00444b6ace4f0c9b7f3b6e047c83e"
OWNERS = {"server-1": "AdaRES-Y94", "server-2": "AdaRES-Y95"}
REPLACEMENTS = {
    "afc93861cf66b0da4835391597ee2d5ce90f6584": OLD_COMMIT,
    "refs/heads/adares-y95-query-set-mean-on-y94": "refs/heads/adares-y96-seed-query-skip-on-y95",
    "refs/heads/adares-y95-query-set-mean": "refs/heads/adares-y96-seed-query-skip",
    "run_y95_bound_native_y94": "run_y96_bound_native_y95",
    "run_y95_native": "run_y96_native",
    "retain_y95_cpu_native": "retain_y96_cpu_native",
    "test_y95_runtime_plan": "test_y96_runtime_plan",
    "rebind_y95_on_y94": "rebind_y96_on_y95",
    "prepare_y95_query_set_mean": "prepare_y96_seed_query_skip",
    "y95_contract": "y96_contract",
    "advance_y94": "advance_y95",
    "y94_deployment": "y95_deployment",
    "y95_y94_additive_binding_": "y96_y95_additive_binding_",
    "original_pre_y94_binding": "original_pre_y95_binding",
    "AdaRES-Y95": "AdaRES-Y96",
    "AdaRES-Y94": "AdaRES-Y95",
    "AdaRES-Y93": "AdaRES-Y94",
    "HEADY94": "HEADY95",
    "Y94/server1": "Y94/server1",
    "Y93/server2": "Y95/server2",
    "Y95 bytes": "Y96 bytes",
    "Y95 additions": "Y96 additions",
    "native-tested Y94": "native-tested Y95",
}
PATTERN = re.compile("|".join(re.escape(key) for key in sorted(REPLACEMENTS, key=len, reverse=True)))

RUNNER = '''
class Runner(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == "daa93b4375f144f0e79da92f85fe1fe27e877e18":
            node.value = Y95
        elif isinstance(node.value, str):
            node.value = node.value.replace("run_y96_native.py", "run_y96_bound_native_y95.py")
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(target, ast.Name) and target.id == "OWNERS" for target in node.targets):
            node.value = ast.parse(repr(OWNERS), mode="eval").body
        return node

    def visit_FunctionDef(self, node):
        if node.name == "validate_binding":
            return ast.parse(BINDING).body[0]
        return self.generic_visit(node)
'''

CURRENT_SOURCE_TEST = '''
def test_actual_parent_native_and_preserved_current_run_sources(self):
    self.assertEqual(parent_evidence(), self.proposal["parent_native"])
    self.assertEqual(list(contract(Y94, Y95, "AdaRES-Y94")), [Y94, Y95])
    queue = monitor.load_queue(str(ROOT))
    for name, cid, pid, version in (
        ("server-1", "AdaRES-Y94", 218513, Y94),
        ("server-2", "AdaRES-Y95", 34484, Y95),
    ):
        task = next(row for row in queue["gpu_queues"][name] if row["id"] == cid)
        self.assertEqual((task["status"], task["pid"], task["code_commit"]), ("running", pid, version))
        self.assertTrue(task["launch_verified"])
        closure.admission(ROOT, task)
'''


class Specialize(ast.NodeTransformer):
    def text(self, value):
        return PATTERN.sub(lambda match: REPLACEMENTS[match.group()], value)

    def visit_Name(self, node):
        node.id = {"Y93": "Y94", "Y94": "Y95"}.get(node.id, node.id)
        return node

    def visit_ImportFrom(self, node):
        node.module = self.text(node.module)
        for alias in node.names:
            alias.name = {"Y93": "Y94", "Y94": "Y95"}.get(alias.name, alias.name)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.text(node.value)
        return node

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == "OWNERS" for target in node.targets):
            node.value = ast.parse(repr(OWNERS), mode="eval").body
            return node
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        if node.name == "Runner":
            return ast.parse(RUNNER).body[0]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "test_actual_parent_native_and_preserved_current_run_sources":
            return ast.parse(CURRENT_SOURCE_TEST).body[0]
        return self.generic_visit(node)


def rebind(token):
    lease(token, 300)
    assert git("rev-parse", "HEAD").decode().strip() == Y95
    source = ROOT / "scripts/rebind_y95_on_y94.py"
    original = ref(source)
    tree = Specialize().visit(ast.parse(source.read_text(encoding="utf-8")))
    body = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    ast.parse(body, feature_version=(3, 8))
    output = ROOT / ("pipeline/checks/y96_y95_binding_wrapper_" + token + ".py")
    assert not output.exists()
    output.write_text(body, encoding="utf-8", newline="\n")
    namespace = {"__name__": "y96_y95_binding_generation", "__file__": __file__}
    exec(compile(body, str(output), "exec"), namespace)
    assert namespace["CID"] == "AdaRES-Y96" and namespace["OLD"] == OLD_COMMIT
    assert namespace["Y94"] == "3f1c351f8a04e12745526e9abb3dbac46af806f1"
    assert namespace["Y95"] == Y95 and namespace["OWNERS"] == OWNERS
    assert namespace["BRANCH"] == "refs/heads/adares-y96-seed-query-skip-on-y95"
    assert namespace["RUNNER"] == "scripts/run_y96_bound_native_y95.py"
    namespace["rebind"](token)
    assert ref(source) == original
    print(json.dumps(dict(wrapper=ref(output), original_rebinding_helper=original,
        scientific_CPU_tests_replayed=False, native_GPU_executed=False, production_HEAD_preserved=Y95)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    rebind(parser.parse_args().token)
