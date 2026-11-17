"""Specialize exact retained Y94 operational helpers after its actual native PASS."""
import argparse
import ast
from datetime import datetime
import importlib
import json
import re
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y93_deployment import Y92, Y93, git
from y94_server1_pre_admission import COMMIT as Y94, PARENT_FREEZE_SHA
import closure


class Specialize(ast.NodeTransformer):
    def __init__(self, deployment):
        self.deployment = deployment
        self.function = None
        self.mapping = ({"Y91": "Y92", "Y92": "Y93", "Y93": "Y94"} if deployment
                        else {"Y90": "Y92", "Y91": "Y93", "Y92": "Y94"})
        self.mapping.update({key.lower(): value.lower() for key, value in list(self.mapping.items())})
        self.pattern = re.compile("|".join(self.mapping))

    def mapped(self, text):
        text = self.pattern.sub(lambda match: self.mapping[match.group()], text)
        text = text.replace("mask_feature_quadratic", "query_geometry_bias")
        text = text.replace("evidence_hidden_skip", "query_geometry_bias")
        text = text.replace("prepare_y94_evidence_hidden_skip", "prepare_y94_query_geometry_bias")
        text = text.replace("run_y94_bound_native.py", "run_y94_bound_native_y93.py")
        return text

    def visit_FunctionDef(self, node):
        old = self.function
        self.function = node.name
        node = self.generic_visit(node)
        self.function = old
        if not self.deployment and node.name == "make_freeze":
            node.body.insert(0, ast.parse(
                "assert parent == tick.load(closure.artifact(ROOT, parentref))").body[0])
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
        if node.module == "run_y92_bound_native":
            node.module = "run_y94_bound_native_y93"
        else:
            node.module = self.mapped(node.module)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if self.deployment and self.function == "require_running_parent_handoff" and node.value == 30:
            node.value = 29
        if isinstance(node.value, str):
            value = self.mapped(node.value)
            if self.deployment and value == Y93:
                value = Y94
            if not self.deployment:
                value = value.replace(
                    "ad60ab5a94db03b12405a043be738aae978786d560d3b707357150994ec95395",
                    PARENT_FREEZE_SHA)
                value = value.replace("static visual mask feature quadratic residual",
                                      "bounded original-query-position self-attention bias")
                value = value.replace("visual-mask quadratic residual",
                                      "bounded original-query-position self-attention bias")
            node.value = value
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if self.deployment and self.function == "observe":
            target = node.targets[0]
            if isinstance(target, ast.Tuple) and [item.id for item in target.elts] == ["name", "commit", "power"]:
                node.value = ast.parse(
                    "{'AdaRES-Y93': ('server-2', Y93, 280.), "
                    "'AdaRES-Y94': ('server-1', Y94, 350.)}[cid]", mode="eval").body
        return node


def create(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y93
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y94")
        assert candidate["stage"] == "validation" and candidate.get("native_smoke_result")
        assert not candidate.get("admission_receipt") and candidate["code_commit"] == Y94
        originals, generated = [], []
        for old, new, deployment in (
            ("advance_y92.py", "advance_y94.py", False),
            ("y93_deployment.py", "y94_deployment.py", True),
        ):
            source, target = ROOT / "scripts" / old, ROOT / "scripts" / new
            assert not target.exists()
            originals.append(ref(source))
            tree = Specialize(deployment).visit(ast.parse(source.read_text(encoding="utf-8")))
            text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
            ast.parse(text, feature_version=(3, 8))
            target.write_text(text, encoding="utf-8", newline="\n")
            generated.append(ref(target))
        advance = importlib.import_module("advance_y94")
        actual, _, _, frozen = advance.prepared(work)
        advance.validate_result(tick.load(closure.artifact(ROOT, actual["native_smoke_result"])), actual, frozen)
        logs = []
        for pattern, count in (("test_y94_execution_guards.py", 10), ("test_y94_admission.py", 19)):
            path = ROOT / ("pipeline/checks/y94_execution_" + token + "_" + pattern[:-3] + ".log")
            assert not path.exists()
            response = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover",
                "-s", "scripts", "-p", pattern, "-v"], cwd=ROOT, capture_output=True, timeout=90)
            path.write_bytes(response.stdout + response.stderr)
            assert response.returncode == 0, str(path)
            assert ("Ran " + str(count) + " tests").encode() in path.read_bytes()
            assert b"skipped" not in path.read_bytes()
            logs.append(ref(path))
        assert all(ref(path) == value for path, value in before.items())
        assert git("ls-files", "--stage", "-z") == index
        assert [ref(entry["path"]) for entry in originals] == originals
        output = ROOT / ("pipeline/checks/y94_execution_preparation_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, token=token, id="AdaRES-Y94", code_commit=Y94,
            observed_at=datetime.now().astimezone().isoformat(), operational_guards=10,
            actual_native_admission_guards=19, actual_native_evidence=candidate["native_smoke_result"],
            source_selected=False, admitted=False, training_launched=False,
            production_HEAD_preserved=Y93, index_queue_running_preserved=True,
            originals_unchanged=True, evidence=[ref(__file__)] + originals + generated + logs
                + [ref("scripts/test_y94_execution_guards.py"), ref("scripts/test_y94_admission.py")]))
        candidate["execution_preparation"] = ref(output)
        candidate["evidence"].extend([ref(output)] + generated + logs)
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(preparation=ref(output), operational_guards=10,
            actual_native_admission_guards=19, source_selected=False, admitted=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    create(parser.parse_args().token)
