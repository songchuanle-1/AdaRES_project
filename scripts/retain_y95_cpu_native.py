"""Retain only the seven CPU-tested Y95 additions on the actual Y93 parent."""
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
from prepare_y95_query_set_mean import CID, files
from y95_contract import validate_cpu
from y93_deployment import Y92, Y93, contract
from retain_y91_cpu_native import git
from prepare_y69_server1_alternative import adapted_body
from y82_server1_alternative import DATA_PROOF
from check_ef_execution import validate_frozen
import closure


HEAD = Y93
BRANCH = "refs/heads/adares-y95-query-set-mean"
OWNERS = {"server-1": "AdaRES-Y92", "server-2": "AdaRES-Y93"}
PAUSE = "pipeline/authorizations/server2_gpu1_pause_20261104.json"


def retain(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    candidate = next(row for row in work["candidates"] if row["id"] == CID)
    assert candidate["stage"] == "implementation" and not candidate.get("code_commit")
    result = ROOT / ("pipeline/analyses/y95_cpu_" + token + ".json")
    validate_cpu(tick.load(result))
    program = ROOT / ("pipeline/checks/y95_cpu_" + token + "_program.py")
    payload = ast.literal_eval(ast.parse(program.read_text(encoding="utf-8")).body[0].value)
    assert payload["goal_sha256"] == tick.load(result)["goal_sha256"] == ref("pipeline/goal.json")["sha256"]
    for path in files():
        value = payload["variant"]["config_data"] if path.suffix == ".yaml" else payload["variant"]["python_files"][path.name]
        assert path.read_bytes() == base64.b64decode(value["base64"]) and ref(path)["sha256"] == value["sha256"]
    assert git("rev-parse", "HEAD").decode().strip() == HEAD
    paths = [path.relative_to(ROOT).as_posix() for path in files()]
    assert not git("ls-files", "--", *paths).strip()
    assert subprocess.run(["git", "show-ref", "--verify", "--quiet", BRANCH], cwd=ROOT).returncode == 1
    before = {path: ref(path) for path in ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
    queue = monitor.load_queue(str(ROOT))
    for name, cid, pid, version in (
        ("server-1", "AdaRES-Y92", 1305468, Y92), ("server-2", "AdaRES-Y93", 47380, Y93),
    ):
        task = next(row for row in queue["gpu_queues"][name] if row["id"] == cid)
        assert task["status"] == "running" and task["launch_verified"] is True and task["pid"] == pid
        assert task["code_commit"] == version
        closure.admission(ROOT, task)
        contract(version, HEAD, cid)
    earlier = copy.deepcopy(next(row for row in work["candidates"] if row["id"] == "AdaRES-Y94"))
    assert earlier["stage"] == "blocked" and earlier["blocker_kind"] == "exclusive_gpu_slot"
    assert not earlier.get("native_smoke_result") and earlier.get("additive_version_binding")
    closure.artifact(ROOT, earlier["source_manifest"])
    base = ROOT / ("pipeline/checks/y95_cpu_version_" + token)
    source, index, output = base.with_name(base.name + "_source_snapshot.json"), base.with_suffix(".index"), base.with_suffix(".json")
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
    commit = git("commit-tree", tree, "-p", HEAD, "-m", "Add independent Y95 all-query mean refinement bypass").decode().strip()
    git("update-ref", BRANCH, commit, "0" * 40)
    assert git("rev-parse", "HEAD").decode().strip() == HEAD
    assert hashlib.sha256(main_index.read_bytes()).hexdigest() == index_sha
    assert all(path.read_bytes() == git("show", commit + ":" + path.relative_to(ROOT).as_posix()) for path in files())
    assert earlier == next(row for row in work["candidates"] if row["id"] == "AdaRES-Y94")
    tick.save(output, dict(
        complete=True, id=CID, code_commit=commit, parent_commit=HEAD, retained_branch=BRANCH,
        working_HEAD=HEAD, main_index_unchanged=True, source_snapshot=ref(source), CPU_result=ref(result),
        CPU_program=ref(program), exact_delta=delta, isolated_index=ref(index), matching_current_owners=OWNERS,
        CPU_tested_scientific_files_exact=True, earlier_Y94_preserved=True,
        native_executed=False, admitted=False, production_sync=False, helper=ref(__file__),
        evidence=[ref(source), ref(result), ref(program), ref(index), ref(__file__)]))
    candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(output),
        additive_version_binding=ref(output), implementation_test_passed=True, CPU_result=ref(result), CPU_program=ref(program))
    candidate["evidence"].extend([ref(output), ref(source), ref(index), ref(result),
                                 ref(result.with_suffix(".log")), ref(__file__)])
    tick.save(ROOT / "pipeline/research_work.json", work)
    assert all(ref(path) == value for path, value in before.items())
    print(json.dumps(dict(version=ref(output), code_commit=commit, HEAD_preserved=HEAD)))


BINDING = '''
def validate_binding(binding, candidate, head):
    assert binding['complete'] is True and binding['id'] == candidate['id'] == 'AdaRES-Y95'
    assert binding['code_commit'] == candidate['code_commit']
    assert binding['parent_commit'] == binding['working_HEAD'] == head == PARENT
    assert binding['matching_current_owners'] == OWNERS
    assert binding['CPU_tested_scientific_files_exact'] is True
    assert binding['main_index_unchanged'] is True and binding['earlier_Y94_preserved'] is True
'''


class Runner(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("run_y94_bound_native_y93.py", "run_y95_native.py")
            node.value = node.value.replace("AdaRES-Y94", CID).replace("y94_", "y95_")
        return node

    def visit_ImportFrom(self, node):
        if node.module == "run_y94_native":
            return None
        node.module = node.module.replace("y94_contract", "y95_contract")
        return node

    def visit_FunctionDef(self, node):
        if node.name == "validate_binding":
            return ast.parse(BINDING).body[0]
        return self.generic_visit(node)

    def visit_keyword(self, node):
        node = self.generic_visit(node)
        if node.arg == "observation" and isinstance(node.value, ast.Constant):
            node.value = ast.Constant("Original native YAML/TRAIN two finite backward/optimizer steps and exact "
                "all-query-mean bypass/original norm-FFN/shared head/hard-gate consumers passed. Efficacy unmeasured.")
        return node


def create_runner():
    original = ROOT / "scripts/run_y94_bound_native_y93.py"
    destination = ROOT / "scripts/run_y95_native.py"
    assert not destination.exists()
    parsed = ast.parse(original.read_text(encoding="utf-8"))
    tree = Runner().visit(parsed)
    template = ast.parse((ROOT / "scripts/run_y94_native.py").read_text(encoding="utf-8"))
    runtime = next(node for node in template.body if isinstance(node, ast.FunctionDef) and node.name == "runtime")
    tree.body.insert(-1, Runner().visit(runtime))
    tree.body[0].value = ast.Constant("Run exact Y95 native task only after matching current GPU0 owner completion.")
    body = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    ast.parse(body, feature_version=(3, 8))
    destination.write_text(body, encoding="utf-8", newline="\n")
    return [ref(original), ref("scripts/run_y94_native.py"), ref(destination)]


def tests(pattern, path, count):
    response = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "scripts",
        "-p", pattern, "-v"], cwd=ROOT, capture_output=True, timeout=60)
    path.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and ("Ran " + str(count) + " tests").encode() in path.read_bytes()
    assert b"skipped" not in path.read_bytes()


def plan(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    candidate = next(row for row in work["candidates"] if row["id"] == CID)
    assert candidate["stage"] == "implementation" and candidate["implementation_test_passed"]
    assert not candidate.get("native_preparation") and git("rev-parse", "HEAD").decode().strip() == HEAD
    version = tick.load(closure.artifact(ROOT, candidate["source_manifest"]))
    payload = tick.load(closure.artifact(ROOT, version["source_snapshot"]))["CPU_tested_payload"]
    validate_cpu(tick.load(closure.artifact(ROOT, candidate["CPU_result"])))
    base = ROOT / ("pipeline/checks/y95_native_" + token)
    first, second, output = base.with_name(base.name + "_guards.log"), base.with_name(base.name + "_runtime_guards.log"), base.with_suffix(".json")
    assert not any(path.exists() for path in (first, second, output))
    tests("test_y95_preparation.py", first, 10)
    authority = ref(PAUSE)
    assert authority["sha256"] == "3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45"
    closure.artifact(ROOT, DATA_PROOF)
    servers = monitor.load_servers(str(ROOT))
    assert servers["server-6"]["allowed_gpus"] == []
    body = closure.artifact(ROOT, candidate["prepared_native_body"]).read_text(encoding="utf-8")
    before = {path: ref(path) for path in ("pipeline/queue.json", "pipeline/running.json")}
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
        program, ppath = base.with_name(base.name + "_" + suffix + "_program.py"), base.with_name(base.name + "_" + suffix + "_plan.json")
        assert not program.exists() and not ppath.exists()
        value = dict(
            complete=True, id=CID, config=candidate["config"], profile="experiment_first",
            code_commit=candidate["code_commit"], goal_sha256=frozen["goal_sha256"], server=name, gpu=0,
            runtime_power_limit_watts=power, resource_authorization=grant, working_HEAD_preserved=HEAD,
            additive_parent=HEAD, retained_branch=BRANCH, native_executed=False, admitted=False, native_PASS_required=True,
            guard_tests_passed=10, Python38_compile=True, execution_helper=runner_refs[-1],
            planned_command="python -B scripts/run_y95_native.py --server " + name + " --token <CURRENT_TOKEN>",
            required_resource="Matching " + OWNERS[name] + " full30/full checkpoint canonical archive and fresh "
                "exclusive idle/unreserved GPU0; earlier Y94 has priority. No GPU1, power/runtime/data substitution "
                "or changed HEAD/owner without a new tested binding.",
            checks="Actual original YAML/sparse TRAIN two finite task losses/backward/updates; six original "
                "self/language tensors, full all-query means, float64 projection references, same original "
                "norm and FFN consumers, hard coarse gates, shared heads, finite new/original/both-input "
                "gradients, zero then learned skip, native modes and memory. CPU is not native PASS.")
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
        blocker="Current Y92/server1 and Y93/server2 occupy both authorized GPU0 resources; earlier Y94 has priority.",
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=["pipeline/queue.json", "pipeline/running.json", "SERVERS.yaml"],
        unblock_condition="Matching original owner natural30/full checkpoint/archive and fresh idle/unreserved "
            "GPU0 with exact tested HEAD/runtime binding; preserve earlier Y94.",
        next_action="Do not replay completed CPU/version/plan checks. At HEADY93 only, after matching Y92/server1 "
            "or Y93/server2 exact full30 checkpoint/archive and fresh authorized idle GPU0, execute "
            "run_y95_native.py after earlier Y94. Further HEAD/owner changes need a new tested binding. "
            "Actual native PASS/full original freeze/admission/exact source selection/safe sync/increasing "
            "native steps remain mandatory. GPU1/Y57 stay paused; no efficacy claim.")
    tick.save(ROOT / "pipeline/research_work.json", work)
    tests("test_y95_runtime_plan.py", second, 8)
    tick.save(output, dict(
        complete=True, id=CID, token=token, code_commit=candidate["code_commit"], CPU_version=candidate["source_manifest"],
        CPU_result=version["CPU_result"], CPU_program=version["CPU_program"], native_plans=plans,
        guard_tests=ref(first), guard_tests_passed=10, runtime_guard_tests=ref(second), runtime_guard_tests_passed=8,
        native_observer_body=candidate["prepared_native_body"], native_observer_callbacks_executed_on_GPU=False,
        same_callbacks_actually_executed_on_CPU=True, original_data_proof=DATA_PROOF,
        original_Y92_Y93_and_earlier_Y94_preserved=True, native_executed=False, admitted=False, production_sync=False,
        helper=ref(__file__), evidence=runner_refs + programs + list(plans.values()) + [ref(first), ref(second)]))
    candidate["native_preparation"] = ref(output)
    candidate["evidence"].extend([ref(output), ref(first), ref(second), authority, DATA_PROOF,
        ref("scripts/y95_contract.py"), ref("scripts/test_y95_preparation.py"),
        ref("scripts/test_y95_runtime_plan.py")] + list(plans.values()) + programs + runner_refs)
    work["tick_attempt"] = dict(
        tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
        candidate_id=CID, kind="implementation_test", result="passed",
        command="python -B scripts/run_experiment_first_probe.py y95_cpu " + version["CPU_program"]["path"]
            + " --token " + token + " --seconds 480",
        observation="Actual original CPU YAML/full model/released backbone/TRAIN loading and scalar/set-peer "
            "causality/seven-stage W2/input-gradient/bypass/two synthetic updates passed, with same native "
            "callbacks executed on CPU;10 actual evidence and8 runtime guards passed. GPU task unexecuted.",
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
