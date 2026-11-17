"""Specialize immutable GPU0 operational helpers for the retained Y93 edge."""
import argparse
import ast
from datetime import datetime
import importlib
import json
import re
import subprocess
import sys

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y92_deployment import Y91, Y92, git
import closure

Y93 = "daa93b4375f144f0e79da92f85fe1fe27e877e18"
PARENT_FREEZE = "e5e6eef716ac892e827fe9131907ec44add8fdad1f619da8a568dacfc6935ca1"


class Specialize(ast.NodeTransformer):
    def __init__(self, name):
        self.name = name
        self.function = None
        mapping = ({"Y90": "Y91", "Y91": "Y92", "Y92": "Y93"}
                   if name == "y93_deployment.py"
                   else {"Y89": "Y91", "Y90": "Y92", "Y91": "Y93"})
        self.mapping = dict(mapping, **{key.lower(): value.lower()
                                      for key, value in mapping.items()})
        self.pattern = re.compile("|".join(self.mapping))

    def mapped(self, value):
        value = self.pattern.sub(lambda match: self.mapping[match.group()], value)
        value = value.replace("mask_feature_quadratic", "evidence_hidden_skip")
        value = value.replace("hidden_confidence_gate", "evidence_hidden_skip")
        value = value.replace("run_y93_bound_native", "run_y93_bound_native_v2")
        value = value.replace("test_y93_execution_guards_v2", "test_y93_execution_guards")
        return value

    def visit_FunctionDef(self, node):
        previous = self.function
        self.function = node.name
        node = self.generic_visit(node)
        self.function = previous
        if self.name == "y93_deployment.py" and node.name == "select":
            for index, statement in enumerate(node.body):
                if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Name)
                        and statement.value.func.id == "validate_result"):
                    node.body.insert(index + 1, ast.parse(
                        "require_running_parent_handoff(work, candidate)").body[0])
                    break
            else:
                raise AssertionError("Missing exact native selection boundary")
        return node

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return node
        value = self.mapped(node.value)
        if self.name == "y93_deployment.py" and value == Y92:
            value = Y93
        if self.name == "advance_y93.py":
            value = value.replace(
                "4ea48fe7d44504047205956a929b32dc18a094319f975707ca5f806c54d2347c",
                PARENT_FREEZE)
            value = value.replace("internal confidence hidden-state gating",
                                  "full normalized-evidence residual into second confidence hidden state")
        if self.name == "test_y93_execution_guards.py":
            if (self.function == "test_reversed_unknown_or_divergent_branches_refused"
                    and value == "AdaRES-Y92"):
                value = "AdaRES-Y94"
        if self.name == "test_y93_admission.py":
            value = value.replace("hidden_gate_calls", "evidence_skip_calls")
            value = value.replace("first_and_second_hidden_identity",
                                  "evidence_and_second_hidden_identity")
            value = value.replace("float64_gate_reference", "float64_residual_reference")
            value = value.replace("final_linear_consumes_gated_hidden",
                                  "final_linear_consumes_residual_hidden")
            value = value.replace("gate_delta_max", "projection_delta_max")
        node.value = value
        return node

    def visit_Name(self, node):
        node.id = self.mapped(node.id)
        return node

    def visit_Attribute(self, node):
        node.attr = self.mapped(node.attr)
        return self.generic_visit(node)

    def visit_alias(self, node):
        node.name = self.mapped(node.name)
        return node

    def visit_ImportFrom(self, node):
        node.module = self.mapped(node.module)
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (self.name == "test_y93_execution_guards.py"
                and self.function == "test_modified_or_unrelated_source_refused"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "validate_delta"
                and isinstance(node.args[0], ast.Name) and node.args[0].id == "rows"):
            node.args[1] = ast.Constant(value="AdaRES-Y94")
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if self.name == "y93_deployment.py" and self.function == "observe":
            target = node.targets[0]
            if (isinstance(target, ast.Tuple)
                    and [item.id for item in target.elts] == ["name", "commit", "power"]):
                node.value = ast.parse(
                    "{'AdaRES-Y92': ('server-1', Y92, 350.), "
                    "'AdaRES-Y93': ('server-2', Y93, 280.)}[cid]", mode="eval").body
        return node


HANDOFF = '''
def require_running_parent_handoff(work, candidate):
    original = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y92")
    assert original["stage"] == "training"
    plan = tick.load(closure.artifact(ROOT, original["terminal_handoff_plan"]))
    assert plan["complete"] is True and plan["id"] == original["id"]
    assert plan["tested_additive_native_evidence"] == candidate["native_smoke_result"]
    assert plan["payload"]["allowed_markers"] == [Y92, Y93]
    assert plan["CPU_guard_tests_passed"] == 30 and plan["additive_guard_tests_passed"] >= 8
    for entry in plan["dependencies"]:
        closure.artifact(ROOT, entry)
'''


def create(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y92
        assert git("rev-parse", Y93 + "^").decode().strip() == Y92
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y93")
        assert candidate["stage"] == "validation" and candidate.get("native_smoke_result")
        assert not candidate.get("admission_receipt")
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        pairs = (
            ("advance_y91.py", "advance_y93.py"),
            ("y92_deployment.py", "y93_deployment.py"),
            ("test_y91_execution_guards.py", "test_y93_execution_guards.py"),
            ("test_y91_admission.py", "test_y93_admission.py"),
        )
        originals = [ref("scripts/" + old) for old, _ in pairs]
        outputs = []
        for old, new in pairs:
            target = ROOT / "scripts" / new
            assert not target.exists()
            parsed = Specialize(new).visit(ast.parse(
                (ROOT / "scripts" / old).read_text(encoding="utf-8")))
            if new == "y93_deployment.py":
                parsed.body.insert(16, ast.parse(HANDOFF).body[0])
            text = ast.unparse(ast.fix_missing_locations(parsed)) + "\n"
            ast.parse(text, feature_version=(3, 8))
            target.write_text(text, encoding="utf-8", newline="\n")
            outputs.append(ref(target))
        advance = importlib.import_module("advance_y93")
        c, _, _, frozen = advance.prepared(work)
        advance.validate_result(tick.load(closure.artifact(ROOT, c["native_smoke_result"])), c, frozen)
        base = ROOT / ("pipeline/checks/y93_execution_preparation_" + token)
        logs = []
        for pattern, count in (
            ("test_y93_execution_guards.py", 13), ("test_y93_admission.py", 22)
        ):
            log = base.with_name(base.name + "_" + pattern[:-3] + ".log")
            assert not log.exists()
            response = subprocess.run(
                [sys.executable, "-B", "-m", "unittest", "discover",
                 "-s", "scripts", "-p", pattern, "-v"],
                cwd=ROOT, capture_output=True, timeout=90)
            log.write_bytes(response.stdout + response.stderr)
            assert response.returncode == 0, str(log)
            assert ("Ran " + str(count) + " tests").encode() in log.read_bytes()
            assert b"skipped" not in log.read_bytes()
            logs.append(ref(log))
        assert [ref(entry["path"]) for entry in originals] == originals
        assert all(ref(path) == entry for path, entry in before.items())
        assert git("ls-files", "--stage", "-z") == index
        output = base.with_suffix(".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, token=token, id=candidate["id"], code_commit=Y93,
            observed_at=datetime.now().astimezone().isoformat(),
            source_resource_protocol_guards=13, actual_native_admission_guards=22,
            actual_native_evidence=candidate["native_smoke_result"],
            evidence=[ref(__file__)] + originals + outputs + logs,
            only_operational_helpers_prepared=True, originals_unchanged=True,
            production_HEAD_preserved=Y92, index_queue_running_preserved=True,
            admitted=False, source_selected=False, training_launched=False))
        candidate["execution_preparation"] = ref(output)
        candidate["evidence"].extend([ref(output)] + outputs + logs)
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(preparation=ref(output), operational_guards=13,
                              actual_native_admission_guards=22, source_selected=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    create(parser.parse_args().token)
