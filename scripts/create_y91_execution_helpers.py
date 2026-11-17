"""Specialize existing operational helpers for retained Y91 on server-2 GPU0."""
import argparse
import ast
from datetime import datetime
import json
import re

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y90_deployment import Y89, Y90, git
import closure

Y91 = "59baf94e9d3e406afa13b2ddc96968cdcc976cae"
SYMBOLS = {"Y88": "Y89", "Y89": "Y90", "Y90": "Y91",
           "y88": "y89", "y89": "y90", "y90": "y91"}
PATTERN = re.compile("|".join(SYMBOLS))
PARENT_FREEZE_SHA = "4ea48fe7d44504047205956a929b32dc18a094319f975707ca5f806c54d2347c"
PAUSE_SHA = "3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45"


def symbols(value):
    return PATTERN.sub(lambda match: SYMBOLS[match.group()], value)


class Specialize(ast.NodeTransformer):
    def __init__(self, name):
        self.name = name
        self.function = None

    def visit_FunctionDef(self, node):
        previous = self.function
        self.function = node.name
        node = self.generic_visit(node)
        self.function = previous
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = symbols(node.value).replace("query_peer_affinity", "hidden_confidence_gate")
            if self.name == "y91_deployment.py" and value == Y90:
                value = Y91
            if self.name == "advance_y91.py":
                value = value.replace("native_smoke_plan_server1", "native_smoke_plan_server2_gpu0")
                value = value.replace("pipeline/authorizations/server1_gpu0_20261025.json",
                                      "pipeline/authorizations/server2_gpu1_pause_20261104.json")
                value = value.replace("bef8248c4136533fc49d612dfe68ab06120c0831660e6dc85e7c7bc0b5694c01",
                                      PAUSE_SHA)
                value = value.replace("93bf51b65a580cbdd5a7ade69fae370d8d0e6e6db5ab665645892c775d84aa81",
                                      PARENT_FREEZE_SHA)
                value = value.replace("--server server-1", "--server server-2")
                value = value.replace("server-1 GPU0", "server-2 GPU0").replace("350W", "280W")
                value = value.replace("independent bounded query-pair weighting of original confidence peers",
                                      "independent internal confidence hidden-state gating")
            if self.name == "prepare_y91_execution.py":
                value = value.replace("server-1 GPU0", "server-2 GPU0")
            if self.name == "test_y91_execution_guards.py":
                if self.function == "test_live_or_incomplete_original_owner_never_unblocks":
                    value = value.replace("server-1", "server-2")
                if self.function == "test_wrong_protocol_lineage_or_resource_refused" and value == "server-2":
                    value = "server-1"
                if self.function == "test_physical_busy_foreign_nonfinite_or_power_change_refused":
                    value = value.replace("350", "POWER_TEMP").replace("280", "350").replace("POWER_TEMP", "280")
                if self.function == "test_reversed_unknown_or_divergent_branches_refused" and value == "AdaRES-Y91":
                    value = "AdaRES-Y92"
            node.value = value
        elif isinstance(node.value, float) and node.value == 350.:
            if self.name == "advance_y91.py" or self.name == "test_y91_execution_guards.py" and self.function in (
                "test_physical_busy_foreign_nonfinite_or_power_change_refused",
                "test_entire_original_review_and_training_preserved",
            ):
                node.value = 280.
        return node

    def visit_Name(self, node):
        node.id = symbols(node.id)
        return node

    def visit_Attribute(self, node):
        node.attr = symbols(node.attr)
        return self.generic_visit(node)

    def visit_alias(self, node):
        node.name = symbols(node.name)
        return node

    def visit_ImportFrom(self, node):
        node.module = symbols(node.module).replace("query_peer_affinity", "hidden_confidence_gate")
        return self.generic_visit(node)

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if self.name == "advance_y91.py" and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Tuple) and [item.id for item in target.elts] == ["NAME", "GPU", "POWER"]:
                node.value = ast.parse("('server-2', 0, 280.)", mode="eval").body
        if self.name == "y91_deployment.py" and self.function == "observe":
            target = node.targets[0]
            if isinstance(target, ast.Tuple) and [item.id for item in target.elts] == ["name", "commit", "power"]:
                node.value = ast.parse(
                    "{'AdaRES-Y90': ('server-1', Y90, 350.), "
                    "'AdaRES-Y91': ('server-2', Y91, 280.)}[cid]", mode="eval").body
        return node


def create(token):
    lease(token, 200)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y90
        assert git("rev-parse", Y91 + "^").decode().strip() == Y90
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y91")
        assert candidate["code_commit"] == Y91 and candidate["stage"] == "blocked"
        assert candidate["blocker_kind"] == "exclusive_gpu_slot"
        assert not candidate.get("native_smoke_result") and not candidate.get("admission_receipt")
        binding = tick.load(closure.artifact(ROOT, candidate["additive_version_binding"]))
        assert binding["code_commit"] == Y91 and binding["parent_commit"] == Y90
        for dependency in binding["evidence"]:
            closure.artifact(ROOT, dependency)
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        names = ("advance_y91.py", "y91_deployment.py",
                 "test_y91_execution_guards.py", "prepare_y91_execution.py")
        outputs = {name: ROOT / "scripts" / name for name in names}
        assert not any(path.exists() for path in outputs.values())
        originals = [ref("scripts/" + name.replace("y91", "y90")) for name in names]
        for name in names:
            source = (ROOT / "scripts" / name.replace("y91", "y90")).read_text(encoding="utf-8")
            tree = Specialize(name).visit(ast.parse(source))
            ast.fix_missing_locations(tree)
            text = ast.unparse(tree) + "\n"
            ast.parse(text, feature_version=(3, 8))
            assert not outputs[name].exists()
            outputs[name].write_text(text, encoding="utf-8", newline="\n")
        assert [ref(ROOT / item["path"]) for item in originals] == originals
        assert all(ref(path) == value for path, value in before.items())
        assert git("ls-files", "--stage", "-z") == index
        output = ROOT / ("pipeline/checks/y91_execution_helpers_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, token=token, id=candidate["id"], code_commit=Y91,
            prepared_at=datetime.now().astimezone().isoformat(),
            server="server-2", physical_gpu=0, power_limit_watts=280.,
            originals=originals, evidence=[ref(__file__)] + [ref(path) for path in outputs.values()],
            original_helpers_unchanged=True, native_executed=False, admitted=False,
            production_HEAD_preserved=Y90, index_queue_running_preserved=True))
        candidate["evidence"].append(ref(output))
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(helpers=ref(output), native_executed=False, admitted=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    create(parser.parse_args().token)
