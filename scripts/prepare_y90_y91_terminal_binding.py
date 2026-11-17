"""Add an explicit tested Y91 source binding without changing the original run."""
import argparse
import ast
import copy
from datetime import datetime
import importlib
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y91_deployment import Y90, Y91, git
import closure


class Specialize(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "y90_terminal_contract":
            node.module = "y90_y91_terminal_contract"
        return self.generic_visit(node)

    def visit_alias(self, node):
        if node.name == "finalize_y90_fixed30":
            node.name = "finalize_y90_fixed30_y91"
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace(
                "test_y90_terminal_handoff.py", "test_y90_terminal_handoff_y91.py").replace(
                "scripts/finalize_y90_fixed30.py", "scripts/finalize_y90_fixed30_y91.py").replace(
                "scripts/y90_terminal_contract.py", "scripts/y90_y91_terminal_contract.py").replace(
                "pipeline/checks/y90_server1_fixed30_", "pipeline/checks/y90_server1_fixed30_y91_")
        return node


def prepare(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y90
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        work = tick.load(ROOT / "pipeline/research_work.json")
        original_work = copy.deepcopy(work)
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y90")
        additive = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y91")
        assert candidate["stage"] == "training" and not candidate.get("terminal_evidence")
        assert additive["stage"] == "validation" and additive.get("native_smoke_result")
        assert not candidate.get("pre_y91_terminal_handoff_plan")
        oldref = candidate["terminal_handoff_plan"]
        old = tick.load(closure.artifact(ROOT, oldref))
        originals = [ref("scripts/" + name) for name in (
            "finalize_y90_fixed30.py", "test_y90_terminal_handoff.py",
            "prepare_y90_terminal_handoff.py", "y90_terminal_contract.py")]
        generated = []
        for original, target in (
            ("finalize_y90_fixed30.py", "finalize_y90_fixed30_y91.py"),
            ("test_y90_terminal_handoff.py", "test_y90_terminal_handoff_y91.py"),
        ):
            path = ROOT / "scripts" / target
            assert not path.exists()
            tree = Specialize().visit(ast.parse((ROOT / "scripts" / original).read_text(encoding="utf-8")))
            ast.fix_missing_locations(tree)
            source = ast.unparse(tree) + "\n"
            ast.parse(source, feature_version=(3, 8))
            path.write_text(source, encoding="utf-8", newline="\n")
            generated.append(ref(path))
        archive = importlib.import_module("finalize_y90_fixed30_y91")
        server, queue, task, _, _ = archive.current()
        payload = copy.deepcopy(old["payload"])
        payload.update(archive.build_contract(task, server))
        assert {key: value for key, value in payload.items()
                if key not in ("allowed_markers", "deployments")} == {
                    key: value for key, value in old["payload"].items()
                    if key not in ("allowed_markers", "deployments")}
        assert set(payload["allowed_markers"]) == {Y90, Y91}
        base = ROOT / ("pipeline/checks/y90_y91_terminal_binding_" + token)
        output = base.with_suffix(".json")
        assert not output.exists()
        logs = []
        for pattern, count, label in (
            ("test_y90_terminal_handoff_y91.py", 25, "lifecycle"),
            ("test_y90_y91_terminal_binding.py", 6, "additive"),
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
        dependencies = old["dependencies"] + generated + [ref(__file__),
            ref("scripts/y90_y91_terminal_contract.py"), ref("scripts/test_y90_y91_terminal_binding.py"),
            ref("scripts/y91_deployment.py"), ref("scripts/y91_contract.py"),
            additive["native_smoke_result"]]
        for dependency in dependencies:
            closure.artifact(ROOT, dependency)
        plan = copy.deepcopy(old)
        plan.update(token=token, observed_at=datetime.now().astimezone().isoformat(),
                    payload=payload, dependencies=dependencies, supersedes=oldref,
                    CPU_guard_tests_passed=25, additive_guard_tests_passed=6,
                    tested_additive_native_evidence=additive["native_smoke_result"],
                    next_command="python -B scripts/finalize_y90_fixed30_y91.py --token <fresh-owned-token>",
                    original_helpers_unchanged=True, original_run_and_protocol_unchanged=True,
                    full_terminal_verification_executed=False, checkpoint_loaded=False,
                    archived=False, training_stopped=False, queue_changed=False, remote_mutations=False,
                    evidence=old["evidence"] + dependencies + logs)
        tick.save(output, plan)
        assert all(ref(path) == value for path, value in before.items())
        assert git("ls-files", "--stage", "-z") == index
        assert [ref(ROOT / item["path"]) for item in originals] == originals
        candidate["pre_y91_terminal_handoff_plan"] = oldref
        candidate["terminal_handoff_plan"] = ref(output)
        candidate["evidence"].extend([ref(output)] + generated + logs + [ref(__file__)])
        assert [row for row in work["candidates"] if row["id"] != "AdaRES-Y90"] == [
            row for row in original_work["candidates"] if row["id"] != "AdaRES-Y90"]
        work["tick_attempt"] = dict(
            tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
            candidate_id="AdaRES-Y90", kind="implementation_test", result="passed",
            command="python -B scripts/prepare_y90_y91_terminal_binding.py --token " + token,
            observation="Same exact original Y90 lifecycle25 plus6 additive source/native/runtime guards passed. "
                        "Original run/checkpoint/receipt/freeze preserved; no checkpoint load or archive.",
            evidence=[ref(output)] + logs)
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(binding=ref(output), lifecycle_guards=25, additive_guards=6,
                              original_helpers_unchanged=True, checkpoint_loaded=False, archived=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    prepare(parser.parse_args().token)
