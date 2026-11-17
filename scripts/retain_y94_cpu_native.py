"""Retain exact CPU-tested additions; prepare only the two authorized GPU0 plans."""
import argparse
import ast
import base64
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y94_query_geometry_bias import CID, files
from y94_contract import validate_cpu
from y92_deployment import Y91, Y92, contract
from retain_y91_cpu_native import git
from prepare_y69_server1_alternative import adapted_body
from y82_server1_alternative import DATA_PROOF
from check_ef_execution import validate_frozen
import closure


HEAD = Y92
BRANCH = "refs/heads/adares-y94-query-geometry-bias"
OWNERS = {"server-1": "AdaRES-Y92", "server-2": "AdaRES-Y91"}
PAUSE = "pipeline/authorizations/server2_gpu1_pause_20261104.json"


def retain(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    candidate = next(row for row in work["candidates"] if row["id"] == CID)
    assert candidate["stage"] == "implementation" and not candidate.get("code_commit")
    result = ROOT / ("pipeline/analyses/y94_cpu_" + token + ".json")
    validate_cpu(tick.load(result))
    program = ROOT / ("pipeline/checks/y94_cpu_" + token + "_program.py")
    payload = ast.literal_eval(ast.parse(program.read_text(encoding="utf-8")).body[0].value)
    assert payload["goal_sha256"] == tick.load(result)["goal_sha256"] == ref("pipeline/goal.json")["sha256"]
    for path in files():
        value = payload["variant"]["config_data"] if path.suffix == ".yaml" else payload["variant"]["python_files"][path.name]
        assert path.read_bytes() == base64.b64decode(value["base64"]) and ref(path)["sha256"] == value["sha256"]
    assert git("rev-parse", "HEAD").decode().strip() == HEAD
    paths = [path.relative_to(ROOT).as_posix() for path in files()]
    assert not git("ls-files", "--", *paths).strip()
    assert subprocess.run(["git", "show-ref", "--verify", "--quiet", BRANCH], cwd=ROOT).returncode == 1
    qref, rref = ref("pipeline/queue.json"), ref("pipeline/running.json")
    queue = monitor.load_queue(str(ROOT))
    for name, cid, pid, version in (
        ("server-1", "AdaRES-Y92", 1305468, Y92), ("server-2", "AdaRES-Y91", 83827, Y91),
    ):
        task = next(row for row in queue["gpu_queues"][name] if row["id"] == cid)
        assert task["status"] == "running" and task["launch_verified"] is True and task["pid"] == pid
        assert task["code_commit"] == version
        closure.admission(ROOT, task)
        contract(version, HEAD, cid)
    earlier = copy.deepcopy(next(row for row in work["candidates"] if row["id"] == "AdaRES-Y93"))
    assert earlier["stage"] == "blocked" and earlier["blocker_kind"] == "exclusive_gpu_slot"
    assert not earlier.get("native_smoke_result") and earlier.get("additive_version_binding")
    closure.artifact(ROOT, earlier["source_manifest"])
    base = ROOT / ("pipeline/checks/y94_cpu_version_" + token)
    source, index, output = (
        base.with_name(base.name + "_source_snapshot.json"), base.with_suffix(".index"), base.with_suffix(".json"))
    assert not any(path.exists() for path in (source, index, output))
    main_index = Path(git("rev-parse", "--git-path", "index").decode().strip())
    if not main_index.is_absolute():
        main_index = ROOT / main_index
    index_sha = hashlib.sha256(main_index.read_bytes()).hexdigest()
    tick.save(source, dict(complete=True, id=CID, token=token, CPU_tested_payload=payload,
        CPU_result=ref(result), CPU_log=ref(result.with_suffix(".log")), CPU_program=ref(program),
        exact_files=[ref(path) for path in files()], working_HEAD=HEAD))
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    git("read-tree", HEAD, env=env)
    for path in files():
        blob = git("hash-object", "-w", "--stdin", data=path.read_bytes()).decode().strip()
        git("update-index", "--add", "--cacheinfo", "100644," + blob + "," + path.relative_to(ROOT).as_posix(), env=env)
    tree = git("write-tree", env=env).decode().strip()
    delta = [row.split("\t") for row in git(
        "diff-tree", "-r", "--no-commit-id", "--name-status", HEAD, tree).decode().splitlines()]
    assert sorted(delta) == sorted([["A", path] for path in paths])
    commit = git("commit-tree", tree, "-p", HEAD, "-m",
        "Add independent Y94 bounded query-self geometry attention bias").decode().strip()
    git("update-ref", BRANCH, commit, "0" * 40)
    assert git("rev-parse", "HEAD").decode().strip() == HEAD
    assert hashlib.sha256(main_index.read_bytes()).hexdigest() == index_sha
    assert all(path.read_bytes() == git("show", commit + ":" + path.relative_to(ROOT).as_posix()) for path in files())
    assert earlier == next(row for row in work["candidates"] if row["id"] == "AdaRES-Y93")
    tick.save(output, dict(
        complete=True, id=CID, code_commit=commit, parent_commit=HEAD, retained_branch=BRANCH,
        working_HEAD=HEAD, main_index_unchanged=True, source_snapshot=ref(source),
        CPU_result=ref(result), CPU_program=ref(program), exact_delta=delta, isolated_index=ref(index),
        earlier_Y93_preserved=True, no_untested_Y93_parent_included=True,
        native_executed=False, admitted=False, production_sync=False, helper=ref(__file__),
        evidence=[ref(source), ref(result), ref(program), ref(index), ref(__file__)]))
    candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(output),
        implementation_test_passed=True, CPU_result=ref(result), CPU_program=ref(program))
    candidate["evidence"].extend([ref(output), ref(source), ref(index), ref(result),
                                 ref(result.with_suffix(".log")), ref(__file__)])
    tick.save(ROOT / "pipeline/research_work.json", work)
    assert (qref, rref) == (ref("pipeline/queue.json"), ref("pipeline/running.json"))
    print(json.dumps(dict(version=ref(output), code_commit=commit, HEAD_preserved=HEAD)))


class RunnerIdentity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("AdaRES-Y93", CID).replace("y93_", "y94_")
            if node.value == Y91:
                node.value = HEAD
        return node

    def visit_ImportFrom(self, node):
        node.module = node.module.replace("y93_contract", "y94_contract")
        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "OWNERS":
            node.value = ast.parse(repr(OWNERS), mode="eval").body
            return node
        return self.generic_visit(node)

    def visit_keyword(self, node):
        node = self.generic_visit(node)
        if node.arg == "observation" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            node.value = ast.Constant(
                "Original native YAML/TRAIN two finite losses/backward/optimizer steps and exact original "
                "sampled positions/query-self geometry/SCG hard-neighborhood consumers passed. Efficacy unmeasured.")
        return node


def create_runner():
    original = ROOT / "scripts/run_y93_native.py"
    destination = ROOT / "scripts/run_y94_native.py"
    assert not destination.exists()
    tree = RunnerIdentity().visit(ast.parse(original.read_text(encoding="utf-8")))
    tree.body[0].value = ast.Constant("Run original Y94 native inputs after exact matching current-owner completion.")
    body = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    ast.parse(body, feature_version=(3, 8))
    assert "binding['parent_commit'] == " + repr(HEAD) in body
    destination.write_text(body, encoding="utf-8", newline="\n")
    return ref(original), ref(destination)


def plan(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    candidate = next(row for row in work["candidates"] if row["id"] == CID)
    assert candidate["stage"] == "implementation" and candidate["implementation_test_passed"]
    assert not candidate.get("native_preparation") and git("rev-parse", "HEAD").decode().strip() == HEAD
    version = tick.load(closure.artifact(ROOT, candidate["source_manifest"]))
    payload = tick.load(closure.artifact(ROOT, version["source_snapshot"]))["CPU_tested_payload"]
    validate_cpu(tick.load(closure.artifact(ROOT, candidate["CPU_result"])))
    assert git("rev-parse", BRANCH).decode().strip() == candidate["code_commit"]
    base = ROOT / ("pipeline/checks/y94_native_" + token)
    tests, runtime_tests, output = (
        base.with_name(base.name + "_guards.log"), base.with_name(base.name + "_runtime_guards.log"), base.with_suffix(".json"))
    assert not any(path.exists() for path in (tests, runtime_tests, output))
    response = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "scripts", "-p", "test_y94_preparation.py", "-v"],
        cwd=ROOT, capture_output=True, timeout=60)
    tests.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and b"Ran 10 tests" in tests.read_bytes() and b"skipped" not in tests.read_bytes()
    authority = ref(PAUSE)
    assert authority["sha256"] == "3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45"
    closure.artifact(ROOT, DATA_PROOF)
    servers = monitor.load_servers(str(ROOT))
    assert servers["server-6"]["allowed_gpus"] == []
    body = closure.artifact(ROOT, candidate["prepared_native_body"]).read_text(encoding="utf-8")
    qref, rref = ref("pipeline/queue.json"), ref("pipeline/running.json")
    runner_refs = create_runner()
    plans, programs = {}, []
    for name, power in (("server-2", 280.), ("server-1", 350.)):
        server = servers[name]
        closure.validate_adares_server(ROOT, server)
        assert server["allowed_gpus"] == [0]
        grant = authority if name == "server-2" else ref("pipeline/authorizations/server1_gpu0_20261025.json")
        frozen = dict(variant=payload["variant"], code_commit=candidate["code_commit"],
            goal_sha256=payload["goal_sha256"], train_source_lf_sha256=payload["train_source_lf_sha256"],
            runtime_server=name, runtime_physical_gpu=0, runtime_power_limit_watts=power, resource_authorization=grant)
        program_body = body
        if name == "server-1":
            frozen["runtime_path_mappings"] = copy.deepcopy(server["runtime_path_mappings"])
            program_body = adapted_body(body, power)
        suffix = name.replace("-", "")
        program = base.with_name(base.name + "_" + suffix + "_program.py")
        ppath = base.with_name(base.name + "_" + suffix + "_plan.json")
        assert not program.exists() and not ppath.exists()
        plan_ = dict(
            complete=True, id=CID, config=candidate["config"], profile="experiment_first",
            code_commit=candidate["code_commit"], goal_sha256=frozen["goal_sha256"], server=name, gpu=0,
            runtime_power_limit_watts=power, resource_authorization=grant, working_HEAD_preserved=HEAD,
            retained_branch=BRANCH, native_executed=False, admitted=False, native_PASS_required=True,
            guard_tests_passed=10, Python38_compile=True, execution_helper=runner_refs[1],
            planned_command="python -B scripts/run_y94_native.py --server " + name + " --token <CURRENT_TOKEN>",
            required_resource="Matching " + OWNERS[name] + " full30 checkpoint verification/canonical archive "
                "and fresh exclusive idle/unreserved authorized GPU0. Earlier Y93 precedes Y94. "
                "No GPU1, power/data/runtime substitution; changed HEAD/current owner needs a NEW binding.",
            checks="Actual original sparse TRAIN two finite task losses/backward/updates; original sampled query "
                "positions, one shared bounded geometry bias consumed by six ordered original self attentions, "
                "exact original SCG hard-neighborhood exclusions, fourteen shared indicators, independent head/"
                "query/expression formula rows, zero then learned bias, finite new/original/query gradients, "
                "native modes and memory. Native callbacks remain unexecuted; CPU is not native PASS.")
        assert validate_frozen(candidate, plan_, frozen) == 7
        program.write_text("FROZEN=" + repr(frozen) + "\n" + program_body, encoding="utf-8", newline="\n")
        ast.parse(program.read_text(encoding="utf-8"), feature_version=(3, 8))
        plan_["program"] = ref(program)
        tick.save(ppath, plan_)
        plans[name] = ref(ppath)
        programs.append(ref(program))
    candidate.update(
        stage="blocked", blocked_from_stage="validation", blocker_kind="exclusive_gpu_slot",
        required_server="server-2", required_gpu=0, resource_work_kind="native_smoke",
        native_smoke_plan=plans["server-2"], native_smoke_plan_server2_gpu0=plans["server-2"],
        native_smoke_plan_server1=plans["server-1"],
        blocker="Current Y92/server1 and Y91/server2 occupy both authorized GPU0 resources; earlier Y93 precedes Y94.",
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=["pipeline/queue.json", "pipeline/running.json", "SERVERS.yaml"],
        unblock_condition="Matching current owner exact natural fixed30 checkpoint/archive, fresh idle/unreserved "
            "GPU0, and current matching HEAD/version/runtime binding. Preserve earlier Y93.",
        next_action="Do not replay completed CPU/version/plan checks. At HEADY92 only, after matching Y92/server1 "
            "orY91/server2 full30 checkpoint/archive and fresh exclusive idle GPU0, run retained run_y94_native.py "
            "after earlier Y93. Further HEAD/owner changes require a NEW tested same-byte/runtime/owner binding. "
            "Actual native PASS/full original freeze/receipt/admission/source selection/safe sync/increasing "
            "steps remain mandatory; GPU1/Y57 stay paused, no efficacy claim.")
    tick.save(ROOT / "pipeline/research_work.json", work)
    response = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "scripts", "-p", "test_y94_runtime_plan.py", "-v"],
        cwd=ROOT, capture_output=True, timeout=60)
    runtime_tests.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and b"Ran 6 tests" in runtime_tests.read_bytes() and b"skipped" not in runtime_tests.read_bytes()
    tick.save(output, dict(
        complete=True, id=CID, token=token, code_commit=candidate["code_commit"],
        CPU_version=candidate["source_manifest"], CPU_result=version["CPU_result"], CPU_program=version["CPU_program"],
        native_plans=plans, guard_tests=ref(tests), guard_tests_passed=10,
        runtime_guard_tests=ref(runtime_tests), runtime_guard_tests_passed=6,
        native_observer_body=candidate["prepared_native_body"], native_observer_callbacks_executed=False,
        original_data_proof=DATA_PROOF, original_Y91_Y92_and_earlier_Y93_preserved=True,
        native_executed=False, admitted=False, production_sync=False, helper=ref(__file__),
        evidence=list(runner_refs) + programs + list(plans.values()) + [ref(tests), ref(runtime_tests)]))
    candidate["native_preparation"] = ref(output)
    candidate["evidence"].extend([ref(output), ref(tests), ref(runtime_tests), authority, DATA_PROOF,
        ref("scripts/y94_contract.py"), ref("scripts/test_y94_preparation.py"),
        ref("scripts/test_y94_runtime_plan.py")] + list(plans.values()) + programs + list(runner_refs))
    work["tick_attempt"] = dict(
        tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
        candidate_id=CID, kind="implementation_test", result="passed",
        command="python -B scripts/run_experiment_first_probe.py y94_cpu " + version["CPU_program"]["path"]
            + " --token " + token + " --seconds 480",
        observation="Actual original CPU YAML/full model/released backbone/TRAIN loading, independent bounded "
            "geometry/custom-self-attention/hard-exclusion/seven-stage W2/input-gradient/causal bypass and two "
            "synthetic updates passed;10 evidence and6 runtime guards passed. Native callbacks and actual "
            "sparse GPU task remain unexecuted, no native PASS/admission/efficacy.",
        evidence=[version["CPU_result"], candidate["source_manifest"], ref(output)])
    tick.save(ROOT / "pipeline/research_work.json", work)
    assert (qref, rref) == (ref("pipeline/queue.json"), ref("pipeline/running.json"))
    print(json.dumps(dict(preparation=ref(output), plans=plans, tests=10, runtime_tests=6,
                          native_executed=False, admitted=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("retain", "plan"))
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    lease(args.token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        {"retain": retain, "plan": plan}[args.operation](args.token)
    finally:
        writer.close()
