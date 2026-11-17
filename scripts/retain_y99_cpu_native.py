"""Retain only tested Y99 additions and prepare GPU-unexecuted native plans."""
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
from prepare_y99_spatial_mask_skip import CID, files
from y99_contract import validate_cpu
from y94_deployment import Y93, Y94, contract
from retain_y91_cpu_native import git
from prepare_y69_server1_alternative import adapted_body
from y82_server1_alternative import DATA_PROOF
from check_ef_execution import validate_frozen
import closure


HEAD = Y94
BRANCH = "refs/heads/adares-y99-spatial-mask-skip"
OWNERS = {"server-1": "AdaRES-Y94", "server-2": "AdaRES-Y93"}
PAUSE = "pipeline/authorizations/server2_gpu1_pause_20261104.json"


def retain(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    candidate = next(row for row in work["candidates"] if row["id"] == CID)
    assert candidate["stage"] == "implementation" and not candidate.get("code_commit")
    result = ROOT / ("pipeline/analyses/y99_cpu_" + token + ".json")
    validate_cpu(tick.load(result))
    program = ROOT / ("pipeline/checks/y99_cpu_" + token + "_program.py")
    payload = ast.literal_eval(ast.parse(program.read_text(encoding="utf-8")).body[0].value)
    assert payload["goal_sha256"] == tick.load(result)["goal_sha256"] == ref("pipeline/goal.json")["sha256"]
    for path in files():
        value = payload["variant"]["config_data"] if path.suffix == ".yaml" else payload["variant"]["python_files"][path.name]
        assert path.read_bytes() == base64.b64decode(value["base64"]) and ref(path)["sha256"] == value["sha256"]
    assert git("rev-parse", "HEAD").decode().strip() == HEAD
    paths = [path.relative_to(ROOT).as_posix() for path in files()]
    assert not git("ls-files", "--", *paths).strip()
    assert subprocess.run(["git", "show-ref", "--verify", "--quiet", BRANCH], cwd=ROOT).returncode == 1
    before = {path: ref(path) for path in
              ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
    others = copy.deepcopy([row for row in work["candidates"] if row["id"] != CID])
    queue = monitor.load_queue(str(ROOT))
    for name, cid, pid, version in (
        ("server-1", "AdaRES-Y94", 218513, Y94), ("server-2", "AdaRES-Y93", 47380, Y93)
    ):
        task = next(row for row in queue["gpu_queues"][name] if row["id"] == cid)
        assert (task["status"], task["launch_verified"], task["pid"], task["code_commit"]) == (
            "running", True, pid, version)
        closure.admission(ROOT, task)
        contract(version, HEAD, cid)
    from advance_y94 import prepared, validate_result
    parent, _, _, frozen = prepared(work)
    actual_native = parent["native_smoke_result"]
    validate_result(tick.load(closure.artifact(ROOT, actual_native)), parent, frozen)
    base = ROOT / ("pipeline/checks/y99_cpu_version_" + token)
    source = base.with_name(base.name + "_source_snapshot.json")
    index, output = base.with_suffix(".index"), base.with_suffix(".json")
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
        git("update-index", "--add", "--cacheinfo",
            "100644," + blob + "," + path.relative_to(ROOT).as_posix(), env=env)
    tree = git("write-tree", env=env).decode().strip()
    delta = [row.split("\t") for row in git(
        "diff-tree", "-r", "--no-commit-id", "--name-status", HEAD, tree).decode().splitlines()]
    assert sorted(delta) == sorted([["A", path] for path in paths])
    commit = git("commit-tree", tree, "-p", HEAD, "-m",
                 "Add independent Y99 full sparse-input static-mask bypass").decode().strip()
    git("update-ref", BRANCH, commit, "0" * 40)
    assert git("rev-parse", "HEAD").decode().strip() == HEAD
    assert hashlib.sha256(main_index.read_bytes()).hexdigest() == index_sha
    assert all(path.read_bytes() == git("show", commit + ":" + path.relative_to(ROOT).as_posix()) for path in files())
    assert [row for row in work["candidates"] if row["id"] != CID] == others
    tick.save(output, dict(
        complete=True, id=CID, code_commit=commit, parent_commit=HEAD, retained_branch=BRANCH,
        working_HEAD=HEAD, main_index_unchanged=True, source_snapshot=ref(source),
        CPU_result=ref(result), CPU_program=ref(program), exact_delta=delta, isolated_index=ref(index),
        matching_current_owners=OWNERS, CPU_tested_scientific_files_exact=True,
        earlier_Y94_preserved=True, earlier_Y95_Y96_Y98_preserved=True,
        actual_parent_native=actual_native, native_executed=False, admitted=False, production_sync=False,
        helper=ref(__file__), evidence=[ref(source), ref(result), ref(program), ref(index), ref(__file__), actual_native]))
    candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(output),
        additive_version_binding=ref(output), implementation_test_passed=True,
        CPU_result=ref(result), CPU_program=ref(program))
    candidate["evidence"].extend([ref(output), ref(source), ref(index), ref(result),
                                 ref(result.with_suffix(".log")), ref(__file__)])
    tick.save(ROOT / "pipeline/research_work.json", work)
    assert all(ref(path) == value for path, value in before.items())
    print(json.dumps(dict(version=ref(output), code_commit=commit, HEAD_preserved=HEAD)))


class Specialize(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == Y93:
            node.value = Y94
        elif isinstance(node.value, str):
            node.value = node.value.replace("Y98", "Y99").replace("y98", "y99")
            node.value = node.value.replace("AdaRES-Y92", "AdaRES-Y94")
        return node

    def visit_ImportFrom(self, node):
        node.module = node.module.replace("y98", "y99")
        return self.generic_visit(node)

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(target, ast.Name) and target.id == "OWNERS" for target in node.targets):
            node.value = ast.parse(repr(OWNERS), mode="eval").body
        return node


def create_runner():
    originals, generated = [], []
    for old, new in (
        ("run_y98_native.py", "run_y99_native.py"),
        ("test_y98_runtime_plan.py", "test_y99_runtime_plan.py"),
    ):
        source, destination = ROOT / "scripts" / old, ROOT / "scripts" / new
        assert not destination.exists()
        originals.append(ref(source))
        tree = Specialize().visit(ast.parse(source.read_text(encoding="utf-8")))
        text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        ast.parse(text, feature_version=(3, 8))
        destination.write_text(text, encoding="utf-8", newline="\n")
        generated.append(ref(destination))
    return originals + generated


def tests(pattern, path, count):
    assert not path.exists()
    response = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover",
        "-s", "scripts", "-p", pattern, "-v"], cwd=ROOT, capture_output=True, timeout=60)
    path.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and ("Ran " + str(count) + " tests").encode() in path.read_bytes(), str(path)
    assert b"skipped" not in path.read_bytes()


def plan(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    candidate = next(row for row in work["candidates"] if row["id"] == CID)
    assert candidate["stage"] == "implementation" and candidate["implementation_test_passed"]
    assert not candidate.get("native_preparation") and git("rev-parse", "HEAD").decode().strip() == HEAD
    version = tick.load(closure.artifact(ROOT, candidate["source_manifest"]))
    payload = tick.load(closure.artifact(ROOT, version["source_snapshot"]))["CPU_tested_payload"]
    validate_cpu(tick.load(closure.artifact(ROOT, candidate["CPU_result"])))
    base = ROOT / ("pipeline/checks/y99_native_" + token)
    first = base.with_name(base.name + "_guards.log")
    second = base.with_name(base.name + "_runtime_guards.log")
    output = base.with_suffix(".json")
    assert not any(path.exists() for path in (first, second, output))
    tests("test_y99_preparation.py", first, 10)
    authority = ref(PAUSE)
    assert authority["sha256"] == "3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45"
    closure.artifact(ROOT, DATA_PROOF)
    servers = monitor.load_servers(str(ROOT))
    assert servers["server-6"]["allowed_gpus"] == []
    body = closure.artifact(ROOT, candidate["prepared_native_body"]).read_text(encoding="utf-8")
    before = {path: ref(path) for path in ("pipeline/queue.json", "pipeline/running.json")}
    runner_refs = create_runner()
    runner_ref = ref("scripts/run_y99_native.py")
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
        value = dict(
            complete=True, id=CID, config=candidate["config"], profile="experiment_first",
            code_commit=candidate["code_commit"], goal_sha256=frozen["goal_sha256"],
            server=name, gpu=0, runtime_power_limit_watts=power, resource_authorization=grant,
            working_HEAD_preserved=HEAD, additive_parent=HEAD, retained_branch=BRANCH,
            native_executed=False, admitted=False, native_PASS_required=True,
            guard_tests_passed=10, Python38_compile=True, execution_helper=runner_ref,
            planned_command="python -B scripts/run_y99_native.py --server " + name + " --token <CURRENT_TOKEN>",
            required_resource="Matching " + OWNERS[name] + " natural30/full finite exact original checkpoint "
                "verification/canonical archive and fresh exclusive idle/unreserved GPU0; earlier Y95/Y96/Y98 "
                "precede Y99. No GPU1, power/runtime/data substitution. Changed HEAD/owner needs NEW tested binding.",
            checks="Actual original YAML/sparse TRAIN two finite task backward/optimizer steps; full32 native "
                "spatial identity, entire original mask MLP and spatial/appearance projections, original "
                "batching/static seven mask consumers, float64 projection references, shared confidence/"
                "query norms/hard-gate algorithm, original/new/raw-spatial gradients, zero then learned skip/"
                "native modes/memory. Actual CPU result is not native PASS.")
        assert validate_frozen(candidate, value, frozen) == 7
        program.write_text("FROZEN=" + repr(frozen) + "\n" + program_body, encoding="utf-8", newline="\n")
        ast.parse(program.read_text(encoding="utf-8"), feature_version=(3, 8))
        value["program"] = ref(program)
        tick.save(ppath, value)
        plans[name] = ref(ppath)
        programs.append(ref(program))
    candidate.update(
        stage="blocked", blocked_from_stage="validation", blocker_kind="exclusive_gpu_slot",
        required_server="server-2", required_gpu=0, resource_work_kind="native_smoke",
        native_smoke_plan=plans["server-2"], native_smoke_plan_server2_gpu0=plans["server-2"],
        native_smoke_plan_server1=plans["server-1"],
        blocker="Original Y94/server1 and Y93/server2 occupy both authorized GPU0 cards; earlier Y95/Y96/Y98 "
            "precede Y99. CPU feasibility is not native sparse TRAIN admission.",
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=["pipeline/queue.json", "pipeline/running.json", "SERVERS.yaml"],
        unblock_condition="Matching original owner natural30/full finite checkpoint/canonical archive and "
            "fresh exclusive idle/unreserved authorized GPU0 with exact tested HEAD/runtime; earlier candidates first.",
        next_action="Do not replay completed CPU/version/plan tests. At HEADY94 ONLY after matching original "
            "Y94/server1 or Y93/server2 natural30/full exact finite checkpoint verification/canonical archive "
            "and fresh exclusive authorized idleGPU0 use run_y99_native.py after earlier Y95/Y96/Y98. "
            "Changed HEAD/owner requires NEW tested same-byte binding. Actual native PASS/full original "
            "freeze/receipt/admission/exact source selection/safe dispatch/increasing counters remain mandatory. "
            "GPU1/Y57 paused; no efficacy claim.")
    tick.save(ROOT / "pipeline/research_work.json", work)
    tests("test_y99_runtime_plan.py", second, 8)
    tick.save(output, dict(
        complete=True, id=CID, token=token, code_commit=candidate["code_commit"],
        CPU_version=candidate["source_manifest"], CPU_result=version["CPU_result"],
        CPU_program=version["CPU_program"], native_plans=plans, guard_tests=ref(first), guard_tests_passed=10,
        runtime_guard_tests=ref(second), runtime_guard_tests_passed=8,
        native_observer_body=candidate["prepared_native_body"], native_observer_callbacks_executed_on_GPU=False,
        same_callbacks_actually_executed_on_CPU=True, original_data_proof=DATA_PROOF,
        original_Y93_Y94_and_earlier_Y95_Y96_Y98_preserved=True,
        native_executed=False, admitted=False, production_sync=False, helper=ref(__file__),
        evidence=runner_refs + programs + list(plans.values()) + [ref(first), ref(second)]))
    candidate["native_preparation"] = ref(output)
    candidate["evidence"].extend([ref(output), ref(first), ref(second), authority, DATA_PROOF,
        ref("scripts/y99_contract.py"), ref("scripts/test_y99_preparation.py"),
        ref("scripts/test_y99_runtime_plan.py")] + list(plans.values()) + programs + runner_refs)
    work["tick_attempt"] = dict(
        tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
        candidate_id=CID, kind="implementation_test", result="passed",
        command="python -B scripts/run_experiment_first_probe.py y99_cpu " + version["CPU_program"]["path"]
            + " --token " + token + " --seconds 480",
        observation="FIRST actual original CPU YAML/fullmodel/releasedbackbone/TRAIN loading plus full32 "
            "spatial scalar/locality/permutation/sign/no-RNG/seven-stage W2/input-gradient/causal bypass "
            "controls and two finite synthetic updates passed; same future native callbacks executed on CPU. "
            "Ten actual-evidence and eight runtime guards passed. GPU native unexecuted; efficacy unknown.",
        evidence=[version["CPU_result"], candidate["source_manifest"], ref(output)])
    tick.save(ROOT / "pipeline/research_work.json", work)
    assert all(ref(path) == value for path, value in before.items())
    print(json.dumps(dict(preparation=ref(output), plans=plans, tests=10, runtime_tests=8,
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
