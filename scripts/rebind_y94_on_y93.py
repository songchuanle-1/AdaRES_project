"""Prepare identical Y94 inputs for the tested Y93 edge without selecting HEAD."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import unittest

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y93_deployment import Y92, Y93, contract, git
from check_ef_execution import validate_frozen
from y94_contract import validate_cpu
from rebind_y90_on_y89 import parse_program, compare_programs, runner_module
import closure

CID = "AdaRES-Y94"
OLD = "8faef574e01ae29bf5504380ed317dcc33420720"
OLD_BRANCH = "refs/heads/adares-y94-query-geometry-bias"
BRANCH = "refs/heads/adares-y94-query-geometry-bias-on-y93"
RUNNER = "scripts/run_y94_bound_native_y93.py"
OWNERS = {"server-1": "AdaRES-Y92", "server-2": "AdaRES-Y93"}
PLAN_KEYS = {"server-1": "native_smoke_plan_server1",
             "server-2": "native_smoke_plan_server2_gpu0"}


class Runner(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("Y93", "Y94").replace("y93", "y94")
            node.value = node.value.replace("run_y94_bound_native_v2", "run_y94_bound_native_y93")
            if node.value == Y92:
                node.value = Y93
        return node

    def visit_ImportFrom(self, node):
        node.module = node.module.replace("y93_contract", "y94_contract")
        if node.module == "run_y93_native":
            node.module = "run_y94_native"
        return self.generic_visit(node)

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(target, ast.Name) and target.id == "OWNERS"
               for target in node.targets):
            node.value = ast.parse(repr(OWNERS), mode="eval").body
        return node


def parent_evidence():
    from advance_y93 import prepared
    from y93_contract import validate_result
    candidate, _, _, frozen = prepared(tick.load(ROOT / "pipeline/research_work.json"))
    entry = candidate["native_smoke_result"]
    validate_result(tick.load(closure.artifact(ROOT, entry)), candidate, frozen)
    return entry


class BindingGuards(unittest.TestCase):
    proposal = None

    def test_exact_only_seven_identical_additions(self):
        p = self.proposal
        self.assertEqual(git("rev-parse", p["candidate"]["code_commit"] + "^").decode().strip(), Y93)
        rows = [row.split("\t") for row in git(
            "diff", "--no-renames", "--name-status", Y93,
            p["candidate"]["code_commit"]).decode().splitlines()]
        self.assertEqual(rows, p["delta"])
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(status == "A" for status, _ in rows))
        self.assertEqual({path for _, path in rows}, set(p["paths"]))
        for path in p["paths"]:
            self.assertEqual((ROOT / path).read_bytes(), git("show", OLD + ":" + path))
            self.assertEqual((ROOT / path).read_bytes(),
                             git("show", p["candidate"]["code_commit"] + ":" + path))

    def test_original_CPU_and_branch_preserved_without_execution(self):
        p = self.proposal
        self.assertEqual(git("rev-parse", OLD_BRANCH).decode().strip(), OLD)
        validate_cpu(tick.load(closure.artifact(ROOT, p["original"]["CPU_result"])))
        for key in ("CPU_result", "CPU_program", "native_preparation", "prepared_native_body"):
            self.assertEqual(p["candidate"][key], p["original"][key])
            closure.artifact(ROOT, p["original"][key])

    def test_both_native_bodies_and_payloads_identical_except_commit(self):
        p = self.proposal
        for name, key in PLAN_KEYS.items():
            old = tick.load(closure.artifact(ROOT, p["original"][key]))
            new = tick.load(closure.artifact(ROOT, p["plans"][name]))
            value = compare_programs(old["program"], new["program"], p["candidate"], OLD)
            self.assertEqual(validate_frozen(p["candidate"], new, value), 7)
            self.assertEqual(new["execution_helper"], ref(RUNNER))

    def test_bad_frozen_source_version_and_goal_refused(self):
        p = self.proposal
        for entry in p["plans"].values():
            plan = tick.load(closure.artifact(ROOT, entry))
            frozen, _ = parse_program(plan["program"])
            for field in ("source", "goal", "commit"):
                changed = copy.deepcopy(frozen)
                if field == "source":
                    changed["variant"]["python_files"]["dec.py"]["base64"] = "YmFk"
                elif field == "goal":
                    changed["goal_sha256"] = "0" * 64
                else:
                    changed["code_commit"] = OLD
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    validate_frozen(p["candidate"], plan, changed)

    def test_original_GPU0_runtimes_and_negative_controls(self):
        p = self.proposal
        module = runner_module(ROOT / RUNNER)
        for name, entry in p["plans"].items():
            plan = tick.load(closure.artifact(ROOT, entry))
            frozen, _ = parse_program(plan["program"])
            server = monitor.load_servers(str(ROOT))[name]
            self.assertEqual(module.runtime(name, p["candidate"], frozen, server),
                             350.0 if name == "server-1" else 280.0)
            for key, value in (("runtime_physical_gpu", 1), ("runtime_server", "server-6"),
                               ("runtime_power_limit_watts", 390.0)):
                with self.assertRaises(AssertionError):
                    module.runtime(name, p["candidate"], dict(frozen, **{key: value}), server)

    def test_current_owner_and_complete_checkpoint_contract(self):
        module = runner_module(ROOT / RUNNER)
        self.assertEqual(module.OWNERS, OWNERS)
        for name, owner in OWNERS.items():
            proof = dict(id=owner, complete=True, process_absent=True, CUDA_initialized=False,
                         completed_epochs={phase: list(range(1, 31)) for phase in ("Train", "Val")},
                         checkpoints=[dict(epoch=epoch, scheduler_last_epoch=epoch,
                                           optimizer_state_count=1, model_finite=True,
                                           optimizer_finite=True) for epoch in (30, 21)])
            module.validate_terminal(proof, name)
            for key, value in (("id", "AdaRES-Y91"), ("process_absent", False),
                               ("CUDA_initialized", True)):
                with self.assertRaises(AssertionError):
                    module.validate_terminal(dict(proof, **{key: value}), name)
            for key, value in (("scheduler_last_epoch", 0), ("optimizer_state_count", 0),
                               ("model_finite", False), ("optimizer_finite", False)):
                changed = copy.deepcopy(proof)
                changed["checkpoints"][0][key] = value
                with self.assertRaises(AssertionError):
                    module.validate_terminal(changed, name)

    def test_future_HEAD_and_matching_binding_are_required(self):
        p = self.proposal
        module = runner_module(ROOT / RUNNER)
        binding = dict(complete=True, id=CID, code_commit=p["candidate"]["code_commit"],
                       parent_commit=Y93, working_HEAD=Y93, matching_current_owners=OWNERS,
                       scientific_files_identical=True, native_program_bodies_identical=True,
                       only_frozen_code_commit_metadata_changed=True)
        module.validate_binding(binding, p["candidate"], Y93)
        for key, value in (("id", "AdaRES-Y93"), ("parent_commit", Y92),
                           ("working_HEAD", Y92), ("scientific_files_identical", False),
                           ("matching_current_owners", {"server-2": "AdaRES-Y91"})):
            with self.assertRaises(AssertionError):
                module.validate_binding(dict(binding, **{key: value}), p["candidate"], Y93)
        with self.assertRaises(AssertionError):
            module.validate_binding(binding, p["candidate"], Y92)

    def test_parent_actual_native_and_additive_sources(self):
        self.assertEqual(parent_evidence(), self.proposal["parent_native"])
        self.assertEqual(list(contract(Y92, Y93, "AdaRES-Y92")), [Y92, Y93])

    def test_production_HEAD_index_queue_and_all_others_preserved(self):
        p = self.proposal
        self.assertEqual(git("rev-parse", "HEAD").decode().strip(), Y92)
        self.assertEqual(hashlib.sha256(Path(p["main_index"]).read_bytes()).hexdigest(),
                         p["main_index_sha256"])
        self.assertEqual(ref("pipeline/queue.json"), p["queue"])
        self.assertEqual(ref("pipeline/running.json"), p["running"])
        work = tick.load(ROOT / "pipeline/research_work.json")
        self.assertEqual([row for row in work["candidates"] if row["id"] != CID],
                         p["other_candidates"])
        for entry in p["original_helpers"]:
            closure.artifact(ROOT, entry)

    def test_resource_pauses_and_pre_mutation_guards(self):
        servers = monitor.load_servers(str(ROOT))
        self.assertEqual(servers["server-1"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-2"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-6"]["allowed_gpus"], [])
        source = (ROOT / RUNNER).read_text(encoding="utf-8")
        self.assertLess(source.index("validate_binding(binding, candidate, closure.git("),
                        source.index("observation = live(server, 0)"))
        self.assertLess(source.index("validate_terminal(terminal, name)"),
                        source.index("observation = live(server, 0)"))
        self.assertIn("assert plan['execution_helper'] == ref(__file__)", source)
        ast.parse(source, feature_version=(3, 8))


def rebind(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y92
        assert subprocess.run(["git", "show-ref", "--verify", "--quiet", BRANCH],
                              cwd=ROOT).returncode == 1
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == CID)
        assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
        assert candidate["code_commit"] == OLD and not candidate.get("native_smoke_result")
        assert not candidate.get("admission_receipt")
        original = copy.deepcopy(candidate)
        from prepare_y94_query_geometry_bias import files
        paths = [path.relative_to(ROOT).as_posix() for path in files()]
        assert len(paths) == 7 and not git("ls-files", "--", *paths).strip()
        base = ROOT / ("pipeline/checks/y94_y93_additive_binding_" + token)
        output, index, log = base.with_suffix(".json"), base.with_suffix(".index"), base.with_suffix(".log")
        assert not any(path.exists() for path in (output, index, log, ROOT / RUNNER))
        main_index = Path(git("rev-parse", "--git-path", "index").decode().strip())
        if not main_index.is_absolute():
            main_index = ROOT / main_index
        index_sha = hashlib.sha256(main_index.read_bytes()).hexdigest()
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        git("read-tree", Y93, env=env)
        for path in paths:
            raw = (ROOT / path).read_bytes()
            assert raw == git("show", OLD + ":" + path)
            blob = git("hash-object", "-w", "--stdin", data=raw).decode().strip()
            git("update-index", "--add", "--cacheinfo", "100644," + blob + "," + path, env=env)
        tree = git("write-tree", env=env).decode().strip()
        commit = git("commit-tree", tree, "-p", Y93, "-m",
                     "Retain identical CPU-tested Y94 files additively on native-tested Y93").decode().strip()
        git("update-ref", BRANCH, commit, "0" * 40)
        parsed = Runner().visit(ast.parse(
            (ROOT / "scripts/run_y93_bound_native_v2.py").read_text(encoding="utf-8")))
        text = ast.unparse(ast.fix_missing_locations(parsed)) + "\n"
        ast.parse(text, feature_version=(3, 8))
        (ROOT / RUNNER).write_text(text, encoding="utf-8", newline="\n")
        proposed = copy.deepcopy(candidate)
        proposed.update(code_commit=commit, retained_branch=BRANCH)
        plans, old_inputs = {}, []
        for name, key in PLAN_KEYS.items():
            oldref = original[key]
            old = tick.load(closure.artifact(ROOT, oldref))
            frozen, body = parse_program(old["program"])
            assert validate_frozen(original, old, frozen) == 7
            frozen["code_commit"] = commit
            program = base.with_name(base.name + "_" + name.replace("-", "") + "_program.py")
            plan_path = base.with_name(base.name + "_" + name.replace("-", "") + "_plan.json")
            assert not program.exists() and not plan_path.exists()
            program.write_text("FROZEN=" + repr(frozen) + "\n" + body,
                               encoding="utf-8", newline="\n")
            plan = copy.deepcopy(old)
            plan.update(code_commit=commit, program=ref(program), retained_branch=BRANCH,
                        supersedes=oldref, additive_parent=Y93, execution_helper=ref(RUNNER),
                        planned_command="python -B " + RUNNER + " --server " + name + " --token <CURRENT>",
                        working_HEAD_preserved=Y92, required_resource=(
                            "At HEADY93 only, after exact matching " + OWNERS[name]
                            + " natural fixed30/full checkpoint archive and fresh idle/unreserved GPU0."))
            tick.save(plan_path, plan)
            compare_programs(old["program"], plan["program"], proposed, OLD)
            plans[name] = ref(plan_path)
            old_inputs.extend([oldref, old["program"]])
        delta = [row.split("\t") for row in git(
            "diff", "--no-renames", "--name-status", Y93, commit).decode().splitlines()]
        proposal = dict(candidate=proposed, original=original, paths=paths, delta=delta,
                        plans=plans, parent_native=parent_evidence(), main_index=str(main_index),
                        main_index_sha256=index_sha, queue=ref("pipeline/queue.json"),
                        running=ref("pipeline/running.json"),
                        other_candidates=[row for row in work["candidates"] if row["id"] != CID],
                        original_helpers=[ref("scripts/run_y94_native.py"),
                                          ref("scripts/run_y93_bound_native_v2.py")])
        BindingGuards.proposal = proposal
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(BindingGuards))
        log.write_text(stream.getvalue(), encoding="utf-8", newline="\n")
        assert result.wasSuccessful() and result.testsRun == 10 and not result.skipped, str(log)
        evidence = [ref(__file__), ref(RUNNER), ref(log), ref(index), proposal["parent_native"]]
        evidence += old_inputs + list(plans.values()) + proposal["original_helpers"]
        evidence += [original[key] for key in ("source_manifest", "CPU_result", "CPU_program",
                                              "native_preparation", "prepared_native_body")]
        evidence += [tick.load(closure.artifact(ROOT, entry))["program"] for entry in plans.values()]
        tick.save(output, dict(
            complete=True, id=CID, token=token, observed_at=datetime.now().astimezone().isoformat(),
            code_commit=commit, parent_commit=Y93, working_HEAD=Y93, prepared_from_HEAD=Y92,
            matching_current_owners=OWNERS, retained_branch=BRANCH, original_commit=OLD,
            scientific_files_identical=True, native_program_bodies_identical=True,
            only_frozen_code_commit_metadata_changed=True, guard_tests_passed=10,
            native_executed=False, admitted=False, scientific_revision=False,
            actual_parent_native=proposal["parent_native"], native_plans=plans,
            original_binding=original["source_manifest"], evidence=evidence))
        candidate["original_pre_y93_binding"] = {key: original[key] for key in (
            "code_commit", "retained_branch", "source_manifest", "native_smoke_plan",
            "native_smoke_plan_server1", "native_smoke_plan_server2_gpu0")}
        candidate.update(
            code_commit=commit, retained_branch=BRANCH, source_manifest=ref(output),
            additive_version_binding=ref(output), native_smoke_plan=plans["server-2"],
            native_smoke_plan_server1=plans["server-1"],
            native_smoke_plan_server2_gpu0=plans["server-2"],
            blocker="Prepared identical Y94 bytes for HEADY93; preserve earlier Y93 and original Y92.",
            unblock_condition="HEADY93, matching Y92/server1 or Y93/server2 full30 checkpoint/archive, then fresh idle/unreserved GPU0.",
            next_action="Do not replay completed CPU/version/binding checks. At HEADY93 use run_y94_bound_native_y93.py only after exact matching Y92/server1 or Y93/server2 natural30/full checkpoint archive and fresh idle GPU0. GPU1/Y57 stay paused.")
        candidate["evidence"].extend([ref(output)] + evidence)
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(binding=ref(output), code_commit=commit, guards=10,
                              native_executed=False, production_HEAD_preserved=Y92)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    rebind(parser.parse_args().token)
