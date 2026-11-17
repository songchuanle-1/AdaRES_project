"""Specialize preserved binding templates; never edit their signed originals."""
import argparse
import ast
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y92_deployment import Y91, Y92, git


OLD = "7a9f1aa67e054d79b28c6af3160a5b1cc11a51c9"
OWNERS = {"server-1": "AdaRES-Y92", "server-2": "AdaRES-Y91"}
PARENT_EVIDENCE = '''
def parent_evidence():
    from advance_y92 import prepared
    from y92_contract import validate_result
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate, _, _, frozen = prepared(work)
    entry = candidate['native_smoke_result']
    validate_result(tick.load(closure.artifact(ROOT, entry)), candidate, frozen)
    task = next(row for row in monitor.load_queue(str(ROOT))['gpu_queues']['server-1']
                if row['id'] == 'AdaRES-Y92')
    assert task['status'] == 'running' and task['launch_verified'] is True
    closure.admission(ROOT, task)
    return [entry, ref(candidate['admission_receipt']), candidate['post_training_freeze']]
'''
PARENT_TEST = '''
def test_additive_parent_has_actual_native_receipt_and_original_frozen_protocol(self):
    entries = parent_evidence()
    self.assertEqual(len(entries), 3)
    for entry in entries:
        closure.artifact(ROOT, entry)
    self.assertEqual(tick.load(closure.artifact(ROOT, entries[0]))['id'], 'AdaRES-Y92')
'''


class Specialize(ast.NodeTransformer):
    def __init__(self, runner):
        self.runner = runner
        self.parent_input_updates = 0
        self.owner_updates = 0

    def visit_ImportFrom(self, node):
        modules = {
            "y92_contract": "y93_contract",
            "run_y92_native": "run_y93_native",
            "y91_deployment": "y92_deployment",
            "prepare_y92_mask_feature_quadratic": "prepare_y93_evidence_hidden_skip",
        }
        node.module = modules.get(node.module, node.module)
        return self.generic_visit(node)

    def visit_alias(self, node):
        if not self.runner:
            node.name = {"Y90": "Y91", "Y91": "Y92"}.get(node.name, node.name)
        return node

    def visit_Name(self, node):
        if not self.runner:
            node.id = {"Y90": "Y91", "Y91": "Y92"}.get(node.id, node.id)
        return node

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            if not self.runner and type(node.value) is int and node.value == 15:
                node.value = 16
            return node
        value = node.value
        exact = {
            "ec332f57dacc09459df8d6d52f21424c4d5ea3de": OLD,
            "refs/heads/adares-y92-mask-feature-quadratic": "refs/heads/adares-y93-evidence-hidden-skip",
            "refs/heads/adares-y92-mask-feature-quadratic-on-y91":
                "refs/heads/adares-y93-evidence-hidden-skip-on-y92",
            "AdaRES-Y93": "AdaRES-Y94",
            "AdaRES-Y92": "AdaRES-Y93",
            "AdaRES-Y91": "AdaRES-Y92",
            "AdaRES-Y90": "AdaRES-Y91",
        }
        if self.runner and value == Y91:
            value = Y92
        else:
            value = exact.get(value, value)
        value = value.replace("run_y92_bound_native.py", "run_y93_bound_native.py")
        value = value.replace("run_y92_native.py", "run_y93_native.py")
        value = value.replace("retain_y92_cpu_native.py", "retain_y93_cpu_native.py")
        value = value.replace("rebind_y92_on_y91.py", "rebind_y93_on_y92.py")
        value = value.replace("pipeline/checks/y92_additive_binding_", "pipeline/checks/y93_additive_binding_")
        value = value.replace("original_pre_y91_binding", "original_pre_y92_binding")
        value = value.replace("y92_", "y93_")
        value = value.replace("CPU-tested Y92", "CPU-tested Y93")
        value = value.replace("on verified Y91", "on verified Y92")
        value = value.replace("Fifteen actual", "Sixteen actual")
        if value.startswith("Original native TRAIN two finite task losses/backward/updates"):
            value = ("Original native YAML/TRAIN two finite task backward and optimizer steps passed "
                     "with full normalized-evidence hidden residual controls. Efficacy unmeasured.")
        if value.startswith("identical seven CPU-tested files on Y91"):
            value = value.replace("on Y91", "on Y92")
        node.value = value
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(target, ast.Name) and target.id == "OWNERS" for target in node.targets):
            node.value = ast.parse(repr(OWNERS), mode="eval").body
            self.owner_updates += 1
        return node

    def visit_AugAssign(self, node):
        node = self.generic_visit(node)
        if not self.runner and isinstance(node.target, ast.Name) and node.target.id == "original_inputs":
            node.value = ast.BinOp(left=node.value, op=ast.Add(),
                                   right=ast.Call(func=ast.Name(id="parent_evidence", ctx=ast.Load()),
                                                  args=[], keywords=[]))
            self.parent_input_updates += 1
        return node

    def visit_keyword(self, node):
        node = self.generic_visit(node)
        if not self.runner and node.arg in ("blocker", "unblock_condition", "next_action"):
            text = {
                "blocker": "Current Y92/server1 and Y91/server2 reserve both authorized GPU0 resources; "
                           "identical CPU-tested Y93 files have a new tested additive binding on Y92.",
                "unblock_condition": "Matching Y92/server1 or Y91/server2 naturally fixed30 complete, "
                                     "exact checkpoint-verified/canonically archived, then matching GPU0 "
                                     "freshly physically idle and unreserved.",
                "next_action": "Do not replay completed CPU/version/binding tests. Run "
                               "scripts/run_y93_bound_native.py --server server-1|server-2 --token CURRENT "
                               "only at HEADY92 after exact matching owner fixed30 checkpoint verification/"
                               "archive and fresh exclusive idle GPU0. Actual native PASS, matching full "
                               "original freeze, admission, safe source selection/sync and increasing steps "
                               "remain required. Further HEAD/owner changes require a NEW tested binding. "
                               "Preserve original GPU1 frozen-review holds and separate Y57 pause.",
            }
            node.value = ast.Constant(value=text[node.arg])
        return node


def create(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y92
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/research_work.json",
            "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        original_refs, generated = [], []
        for source_name, target_name, runner in (
            ("run_y92_bound_native.py", "run_y93_bound_native.py", True),
            ("rebind_y92_on_y91.py", "rebind_y93_on_y92.py", False),
        ):
            source = ROOT / "scripts" / source_name
            target = ROOT / "scripts" / target_name
            assert not target.exists()
            original_refs.append(ref(source))
            transform = Specialize(runner)
            tree = transform.visit(ast.parse(source.read_text(encoding="utf-8")))
            assert transform.owner_updates == 1
            assert transform.parent_input_updates == (0 if runner else 1)
            if not runner:
                helper = ast.parse(PARENT_EVIDENCE).body[0]
                position = next(i for i, node in enumerate(tree.body)
                                if isinstance(node, ast.ClassDef) and node.name == "BindingGuards")
                tree.body.insert(position, helper)
                tests = next(node for node in tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "BindingGuards")
                tests.body.append(ast.parse(PARENT_TEST).body[0])
                assert sum(isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                           for node in tests.body) == 16
            ast.fix_missing_locations(tree)
            body = ast.unparse(tree) + "\n"
            ast.parse(body, feature_version=(3, 8))
            target.write_text(body, encoding="utf-8", newline="\n")
            generated.append(ref(target))
        assert all(ref(path) == entry for path, entry in before.items())
        assert git("ls-files", "--stage", "-z") == index
        assert [ref(ROOT / entry["path"]) for entry in original_refs] == original_refs
        output = ROOT / ("pipeline/checks/y93_y92_binding_helpers_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, token=token, observed_at=__import__("datetime").datetime.now().astimezone().isoformat(),
            original_templates=original_refs, generated=generated, helper=ref(__file__),
            generation_only=True, guards_executed=False, native_executed=False,
            production_HEAD_index_queue_and_work_unchanged=True))
        print(json.dumps(dict(manifest=ref(output), generated=generated, guards_executed=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    create(parser.parse_args().token)
