"""Bind unchanged original Y91 terminal checks to the actually tested Y92 edge."""
import argparse
import ast
import copy
from datetime import datetime
import importlib
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y92_deployment import Y91, Y92, git
from y91_y92_terminal_contract import native_evidence
import closure


class Specialize(ast.NodeTransformer):
    def __init__(self):
        self.expectation_updates = 0

    def visit_ImportFrom(self, node):
        if node.module == "y91_terminal_contract":
            node.module = "y91_y92_terminal_contract"
        return self.generic_visit(node)

    def visit_alias(self, node):
        if node.name == "finalize_y91_fixed30":
            node.name = "finalize_y91_fixed30_y92"
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace(
                "test_y91_terminal_handoff.py", "test_y91_terminal_handoff_y92.py").replace(
                "scripts/finalize_y91_fixed30.py", "scripts/finalize_y91_fixed30_y92.py").replace(
                "scripts/y91_terminal_contract.py", "scripts/y91_y92_terminal_contract.py").replace(
                "pipeline/checks/y91_server2_fixed30_", "pipeline/checks/y91_server2_fixed30_y92_")
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "assertEqual"
                and len(node.args) == 2 and isinstance(node.args[1], ast.List)
                and len(node.args[1].elts) == 1
                and isinstance(node.args[1].elts[0], ast.Name)
                and node.args[1].elts[0].id == "Y91"):
            assert isinstance(node.args[0], ast.Subscript)
            assert isinstance(node.args[0].slice, ast.Constant)
            assert node.args[0].slice.value == "allowed_markers"
            node.args[1].elts.append(ast.Constant(value=Y92))
            self.expectation_updates += 1
        return node


def prepare(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y91
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml",
            "pipeline/admissions/AdaRES-Y91.json")}
        index = git("ls-files", "--stage", "-z")
        work = tick.load(ROOT / "pipeline/research_work.json")
        original_work = copy.deepcopy(work)
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y91")
        additive = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y92")
        assert candidate["stage"] == "training" and not candidate.get("terminal_evidence")
        assert additive["stage"] == "validation" and not additive.get("admission_receipt")
        actual_native = native_evidence(work)
        assert not candidate.get("pre_y92_terminal_handoff_plan")
        oldref = candidate["terminal_handoff_plan"]
        old = tick.load(closure.artifact(ROOT, oldref))
        originals = [ref("scripts/" + name) for name in (
            "finalize_y91_fixed30.py", "test_y91_terminal_handoff.py",
            "prepare_y91_terminal_handoff.py", "y91_terminal_contract.py")]
        generated = []
        for original, target, count in (
            ("finalize_y91_fixed30.py", "finalize_y91_fixed30_y92.py", 0),
            ("test_y91_terminal_handoff.py", "test_y91_terminal_handoff_y92.py", 1),
        ):
            path = ROOT / "scripts" / target
            assert not path.exists()
            transform = Specialize()
            tree = transform.visit(ast.parse((ROOT / "scripts" / original).read_text(encoding="utf-8")))
            assert transform.expectation_updates == count
            ast.fix_missing_locations(tree)
            source = ast.unparse(tree) + "\n"
            ast.parse(source, feature_version=(3, 8))
            path.write_text(source, encoding="utf-8", newline="\n")
            generated.append(ref(path))
        archive = importlib.import_module("finalize_y91_fixed30_y92")
        server, queue, task, _, _ = archive.current()
        payload = copy.deepcopy(old["payload"])
        payload.update(archive.build_contract(task, server))
        assert {key: value for key, value in payload.items()
                if key not in ("allowed_markers", "deployments")} == {
                    key: value for key, value in old["payload"].items()
                    if key not in ("allowed_markers", "deployments")}
        assert payload["allowed_markers"] == [Y91, Y92]
        base = ROOT / ("pipeline/checks/y91_y92_terminal_binding_" + token)
        output = base.with_suffix(".json")
        assert not output.exists()
        logs = []
        for pattern, count, label in (
            ("test_y91_terminal_handoff_y92.py", 29, "lifecycle"),
            ("test_y91_y92_terminal_binding.py", 8, "additive"),
        ):
            path = base.with_name(base.name + "_" + label + ".log")
            assert not path.exists()
            response = subprocess.run(
                [sys.executable, "-B", "-m", "unittest", "discover", "-s", "scripts", "-p", pattern, "-v"],
                cwd=ROOT, capture_output=True, timeout=90)
            path.write_bytes(response.stdout + response.stderr)
            assert response.returncode == 0 and ("Ran " + str(count) + " tests").encode() in path.read_bytes()
            assert b"skipped" not in path.read_bytes()
            logs.append(ref(path))
        dependencies = old["dependencies"] + generated + [
            ref(__file__), ref("scripts/y91_y92_terminal_contract.py"),
            ref("scripts/test_y91_y92_terminal_binding.py"), ref("scripts/y92_deployment.py"),
            ref("scripts/y92_contract.py"), ref("scripts/advance_y92.py"),
            actual_native, additive["additive_version_binding"], additive["native_smoke_plan"]]
        for dependency in dependencies:
            closure.artifact(ROOT, dependency)
        plan = copy.deepcopy(old)
        plan.update(
            token=token, observed_at=datetime.now().astimezone().isoformat(),
            payload=payload, dependencies=dependencies, supersedes=oldref,
            CPU_guard_tests_passed=29, additive_guard_tests_passed=8,
            tested_additive_native_evidence=actual_native,
            next_command="python -B scripts/finalize_y91_fixed30_y92.py --token <fresh-owned-token>",
            original_helpers_unchanged=True, original_run_and_protocol_unchanged=True,
            full_terminal_verification_executed=False, checkpoint_loaded=False,
            archived=False, training_stopped=False, queue_changed=False, remote_mutations=False,
            evidence=old["evidence"] + dependencies + logs)
        tick.save(output, plan)
        assert all(ref(path) == entry for path, entry in before.items())
        assert git("ls-files", "--stage", "-z") == index
        assert [ref(ROOT / entry["path"]) for entry in originals] == originals
        candidate["pre_y92_terminal_handoff_plan"] = oldref
        candidate["terminal_handoff_plan"] = ref(output)
        candidate["evidence"].extend([ref(output)] + generated + logs + [
            ref(__file__), ref("scripts/y91_y92_terminal_contract.py"),
            ref("scripts/test_y91_y92_terminal_binding.py")])
        assert [row for row in work["candidates"] if row["id"] != candidate["id"]] == [
            row for row in original_work["candidates"] if row["id"] != candidate["id"]]
        work["tick_attempt"] = dict(
            tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
            candidate_id=candidate["id"], kind="implementation_test", result="passed",
            command="python -B scripts/prepare_y91_y92_terminal_binding.py --token " + token,
            observation="Same original Y91 lifecycle29 plus8 actual additive source/native/runtime guards passed. "
                        "Original PID/checkpoint/receipt/freeze retained; no Y91 checkpoint load or archive.",
            evidence=[ref(output)] + logs)
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(binding=ref(output), lifecycle_guards=29, additive_guards=8,
                              original_helpers_unchanged=True, checkpoint_loaded=False, archived=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    prepare(parser.parse_args().token)
