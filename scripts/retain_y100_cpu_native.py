"""Retain exact CPU-tested Y100 files in isolation and freeze unexecuted native tasks."""
import argparse
import ast
import base64
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y100_appearance_input_skip import CID, files
from y100_contract import validate_cpu
from y99_deployment_v2 import Y98, Y99, contract
from retain_y91_cpu_native import git
from prepare_y69_server1_alternative import adapted_body
from y82_server1_alternative import DATA_PROOF
from check_ef_execution import validate_frozen
import closure

HEAD = Y99
BRANCH = "refs/heads/adares-y100-appearance-input-skip"
OWNERS = {"server-1": "AdaRES-Y99", "server-2": "AdaRES-Y98"}
PAUSE = "pipeline/authorizations/server2_gpu1_pause_20261104.json"


def tests(pattern, path, count):
    assert not path.exists()
    response = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "scripts", "-p", pattern, "-v"], cwd=ROOT, capture_output=True, timeout=60)
    path.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and ("Ran " + str(count) + " tests").encode() in path.read_bytes(), str(path)
    assert b"skipped" not in path.read_bytes()


def retain(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    candidate = next(r for r in work["candidates"] if r["id"] == CID)
    assert candidate["stage"] == "implementation" and not candidate.get("code_commit")
    result = ROOT / ("pipeline/analyses/y100_cpu_" + token + ".json")
    validate_cpu(tick.load(result))
    guards = ROOT / ("pipeline/checks/y100_CPU_guards_" + token + ".log")
    tests("test_y100_preparation.py", guards, 10)
    program = ROOT / ("pipeline/checks/y100_cpu_" + token + "_program.py")
    payload = ast.literal_eval(ast.parse(program.read_text(encoding="utf-8")).body[0].value)
    assert payload["goal_sha256"] == tick.load(result)["goal_sha256"] == ref("pipeline/goal.json")["sha256"]
    for p in files():
        value = payload["variant"]["config_data"] if p.suffix == ".yaml" else payload["variant"]["python_files"][p.name]
        assert p.read_bytes() == base64.b64decode(value["base64"]) and ref(p)["sha256"] == value["sha256"]
    assert git("rev-parse", "HEAD").decode().strip() == HEAD
    paths = [p.relative_to(ROOT).as_posix() for p in files()]
    assert not git("ls-files", "--", *paths).strip()
    assert subprocess.run(["git", "show-ref", "--verify", "--quiet", BRANCH], cwd=ROOT).returncode == 1
    before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
    queue = monitor.load_queue(str(ROOT))
    for name, cid, pid, commit in (("server-1", "AdaRES-Y99", 2980836, Y99), ("server-2", "AdaRES-Y98", 25962, Y98)):
        task = next(r for r in queue["gpu_queues"][name] if r["id"] == cid)
        assert (task["status"], task["launch_verified"], task["pid"], task["code_commit"]) == ("running", True, pid, commit)
        closure.admission(ROOT, task)
        contract(commit, HEAD, cid)
    from advance_y99_v2 import prepared, validate_result
    parent, _, _, frozen = prepared(work)
    native = parent["native_smoke_result"]
    validate_result(tick.load(closure.artifact(ROOT, native)), parent, frozen)
    base = ROOT / ("pipeline/checks/y100_cpu_version_" + token)
    source = base.with_name(base.name + "_source_snapshot.json")
    index, output = base.with_suffix(".index"), base.with_suffix(".json")
    assert not any(p.exists() for p in (source, index, output))
    main_index = Path(git("rev-parse", "--git-path", "index").decode().strip())
    if not main_index.is_absolute():
        main_index = ROOT / main_index
    index_sha = hashlib.sha256(main_index.read_bytes()).hexdigest()
    tick.save(source, dict(complete=True, id=CID, token=token, CPU_tested_payload=payload, CPU_result=ref(result),
        CPU_log=ref(result.with_suffix(".log")), CPU_program=ref(program), exact_files=[ref(p) for p in files()], working_HEAD=HEAD))
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    git("read-tree", HEAD, env=env)
    for p in files():
        blob = git("hash-object", "-w", "--stdin", data=p.read_bytes()).decode().strip()
        git("update-index", "--add", "--cacheinfo", "100644," + blob + "," + p.relative_to(ROOT).as_posix(), env=env)
    tree = git("write-tree", env=env).decode().strip()
    delta = [r.split("\t") for r in git("diff-tree", "-r", "--no-commit-id", "--name-status", HEAD, tree).decode().splitlines()]
    assert sorted(delta) == sorted([["A", p] for p in paths])
    commit = git("commit-tree", tree, "-p", HEAD, "-m", "Add independent Y100 full appearance input-projection bypass").decode().strip()
    git("update-ref", BRANCH, commit, "0" * 40)
    assert git("rev-parse", "HEAD").decode().strip() == HEAD and hashlib.sha256(main_index.read_bytes()).hexdigest() == index_sha
    assert all(p.read_bytes() == git("show", commit + ":" + p.relative_to(ROOT).as_posix()) for p in files())
    tick.save(output, dict(complete=True, id=CID, code_commit=commit, parent_commit=HEAD, retained_branch=BRANCH,
        working_HEAD=HEAD, main_index_unchanged=True, CPU_tested_scientific_files_exact=True,
        source_snapshot=ref(source), CPU_result=ref(result), CPU_program=ref(program), exact_delta=delta,
        isolated_index=ref(index), matching_current_owners=OWNERS, actual_parent_native=native,
        native_executed=False, admitted=False, production_sync=False, helper=ref(__file__),
        evidence=[ref(source), ref(result), ref(result.with_suffix(".log")), ref(program), ref(index), ref(guards), ref(__file__), native]))
    candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(output), additive_version_binding=ref(output),
        implementation_test_passed=True, CPU_result=ref(result), CPU_program=ref(program), CPU_guard_tests=ref(guards))
    candidate["evidence"].extend([ref(output), ref(source), ref(index), ref(result), ref(result.with_suffix(".log")), ref(guards), ref(__file__)])
    tick.save(ROOT / "pipeline/research_work.json", work)
    assert all(ref(p) == r for p, r in before.items())
    print(json.dumps(dict(version=ref(output), code_commit=commit, HEAD_preserved=HEAD, CPU_guards=10)))


class Rewrite(ast.NodeTransformer):
    def __init__(self, replacements):
        self.replacements = replacements
        self.pattern = re.compile("|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)))
    def text(self, value):
        return self.pattern.sub(lambda m: self.replacements[m.group()], value)
    def visit_Constant(self, n):
        if isinstance(n.value, str):
            n.value = self.text(n.value)
        return n
    def visit_Name(self, n):
        n.id = self.text(n.id)
        return n
    def visit_Attribute(self, n):
        n.attr = self.text(n.attr)
        return self.generic_visit(n)
    def visit_ImportFrom(self, n):
        n.module = self.text(n.module)
        return self.generic_visit(n)
    def visit_alias(self, n):
        n.name = self.text(n.name)
        return n


def create_runner():
    source = ROOT / "scripts/run_y99_native.py"
    target = ROOT / "scripts/run_y100_native.py"
    assert not target.exists()
    changes = {"Y99": "Y100", "y99": "y100", "AdaRES-Y94": "AdaRES-Y99", "AdaRES-Y93": "AdaRES-Y98"}
    tree = Rewrite(changes).visit(ast.parse(source.read_text(encoding="utf-8")))
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARENT" for t in n.targets):
            n.value = ast.Constant(value=HEAD)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OWNERS" for t in n.targets):
            n.value = ast.parse(repr(OWNERS), mode="eval").body
    text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    text = text.replace("Original native YAML/TRAIN two finite backward/optimizer steps and exact full1024 appearance/original mask base/batching/static-mask/shared-head/hard-gate consumers passed. Efficacy unmeasured.",
        "Original native YAML/TRAIN two finite backward/optimizer steps and exact full1024 appearance/original projection/sum_norm/visual-memory/static-mask/shared-head/hard-gate consumers passed. Efficacy unmeasured.")
    ast.parse(text, feature_version=(3, 8))
    target.write_text(text, encoding="utf-8", newline="\n")
    old = ROOT / "scripts/test_y99_runtime_plan.py"
    new = ROOT / "scripts/test_y100_runtime_plan.py"
    assert not new.exists()
    body = ast.unparse(ast.fix_missing_locations(Rewrite(changes).visit(ast.parse(old.read_text(encoding="utf-8"))))) + "\n"
    ast.parse(body, feature_version=(3, 8))
    new.write_text(body, encoding="utf-8", newline="\n")
    return [ref(source), ref(target), ref(old), ref(new)]


def plan(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    c = next(r for r in work["candidates"] if r["id"] == CID)
    assert c["stage"] == "implementation" and c["implementation_test_passed"] and not c.get("native_preparation")
    assert git("rev-parse", "HEAD").decode().strip() == HEAD
    version = tick.load(closure.artifact(ROOT, c["source_manifest"]))
    payload = tick.load(closure.artifact(ROOT, version["source_snapshot"]))["CPU_tested_payload"]
    validate_cpu(tick.load(closure.artifact(ROOT, c["CPU_result"])))
    base = ROOT / ("pipeline/checks/y100_native_" + token)
    output, guards = base.with_suffix(".json"), base.with_name(base.name + "_runtime_guards.log")
    assert not output.exists() and not guards.exists()
    authority = ref(PAUSE)
    assert authority["sha256"] == "3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45"
    closure.artifact(ROOT, DATA_PROOF)
    servers = monitor.load_servers(str(ROOT))
    assert servers["server-6"]["allowed_gpus"] == []
    body = closure.artifact(ROOT, c["prepared_native_body"]).read_text(encoding="utf-8")
    before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json")}
    runner_refs = create_runner()
    plans, programs = {}, []
    for name, power in (("server-2", 280.), ("server-1", 350.)):
        server = servers[name]
        closure.validate_adares_server(ROOT, server)
        assert server["allowed_gpus"] == [0]
        grant = authority if name == "server-2" else ref("pipeline/authorizations/server1_gpu0_20261025.json")
        frozen = dict(variant=payload["variant"], code_commit=c["code_commit"], goal_sha256=payload["goal_sha256"],
            train_source_lf_sha256=payload["train_source_lf_sha256"], runtime_server=name, runtime_physical_gpu=0,
            runtime_power_limit_watts=power, resource_authorization=grant)
        program_body = body
        if name == "server-1":
            frozen["runtime_path_mappings"] = copy.deepcopy(server["runtime_path_mappings"])
            program_body = adapted_body(body, power)
        suffix = name.replace("-", "")
        program = base.with_name(base.name + "_" + suffix + "_program.py")
        ppath = base.with_name(base.name + "_" + suffix + "_plan.json")
        assert not program.exists() and not ppath.exists()
        value = dict(complete=True, id=CID, config=c["config"], profile="experiment_first", code_commit=c["code_commit"],
            goal_sha256=frozen["goal_sha256"], server=name, gpu=0, runtime_power_limit_watts=power, resource_authorization=grant,
            working_HEAD_preserved=HEAD, additive_parent=HEAD, retained_branch=BRANCH, native_executed=False,
            admitted=False, native_PASS_required=True, guard_tests_passed=10, Python38_compile=True,
            execution_helper=ref("scripts/run_y100_native.py"),
            planned_command="python -B scripts/run_y100_native.py --server " + name + " --token <CURRENT_TOKEN>",
            required_resource="Matching original " + OWNERS[name] + " natural fixed30/full finite exact original checkpoint verification/canonical archive, fresh exclusive idle/unreserved GPU0, exact HEAD/private runtime/data/mappings/350W280W. No GPU1 or resource substitution.",
            checks="Actual original YAML/sparse TRAIN two finite backward/optimizer steps, full1024 raw identity/original projection/sum_norm/first visual memory/static seven mask readouts/shared confidence and original gates/new+original gradients/native modes/budget. CPU feasibility is not native PASS.")
        assert validate_frozen(c, value, frozen) == 7
        program.write_text("FROZEN=" + repr(frozen) + "\n" + program_body, encoding="utf-8", newline="\n")
        ast.parse(program.read_text(encoding="utf-8"), feature_version=(3, 8))
        value["program"] = ref(program)
        tick.save(ppath, value)
        plans[name] = ref(ppath)
        programs.append(ref(program))
    c.update(stage="blocked", blocked_from_stage="validation", blocker_kind="exclusive_gpu_slot",
        required_server="server-2", required_gpu=0, resource_work_kind="native_smoke",
        native_smoke_plan=plans["server-2"], native_smoke_plan_server2_gpu0=plans["server-2"], native_smoke_plan_server1=plans["server-1"],
        blocker="Original Y98/server2GPU0 and Y99/server1GPU0 remain running. First Y100 CPU feasibility passed; native sparse TRAIN remains unexecuted and is required for admission.",
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=["pipeline/queue.json", "pipeline/running.json", "SERVERS.yaml"],
        unblock_condition="Matching original Y98/server2 or Y99/server1 natural30/full finite original checkpoint verification/canonical archive plus fresh authorized exclusive idle/unreserved GPU0 and exact tested HEAD/runtime.",
        next_action="Do not replay CPU/version/plan tests. At HEADY99 and matching original Y98/server2 or Y99/server1 natural fixed30/full exact checkpoint/canonical archive use run_y100_native.py on fresh exclusive idle authorized GPU0. Changed HEAD/owner requires new tested binding. Actual native PASS then full original selected-device frozen protocol/receipt/admission/source selection/safe dispatch/increasing native counters in same tick; no GPU1/Y57 use, efficacy unknown.")
    tick.save(ROOT / "pipeline/research_work.json", work)
    tests("test_y100_runtime_plan.py", guards, 8)
    tick.save(output, dict(complete=True, id=CID, token=token, code_commit=c["code_commit"], CPU_version=c["source_manifest"],
        CPU_result=c["CPU_result"], CPU_program=c["CPU_program"], native_plans=plans, guard_tests=c["CPU_guard_tests"], guard_tests_passed=10,
        runtime_guard_tests=ref(guards), runtime_guard_tests_passed=8, native_observer_body=c["prepared_native_body"],
        native_observer_callbacks_executed_on_GPU=False, same_callbacks_actually_executed_on_CPU=True, original_data_proof=DATA_PROOF,
        original_Y98_Y99_preserved=True, native_executed=False, admitted=False, production_sync=False, helper=ref(__file__),
        evidence=runner_refs + programs + list(plans.values()) + [ref(guards), c["CPU_guard_tests"]]))
    c["native_preparation"] = ref(output)
    c["evidence"].extend([ref(output), ref(guards), authority, DATA_PROOF, ref("scripts/y100_contract.py"), ref("scripts/test_y100_preparation.py"), ref("scripts/test_y100_runtime_plan.py")] + list(plans.values()) + programs + runner_refs)
    work["tick_attempt"] = dict(tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
        candidate_id=CID, kind="implementation_test", result="passed",
        command="python -B scripts/run_experiment_first_probe.py y100_cpu " + c["CPU_program"]["path"] + " --token " + token + " --seconds 600",
        observation="FIRST original CPU YAML/fullmodel/releasedbackbone/TRAIN collation, full raw appearance scalar/locality/permutation/sign/no-RNG/seven-stage W2 anchors/input gradients/causal bypass and two finite synthetic updates passed. Same future native callbacks executed on CPU; ten actual-evidence and eight runtime guards passed. Native GPU unexecuted; efficacy unknown.",
        evidence=[c["CPU_result"], c["source_manifest"], ref(output)])
    tick.save(ROOT / "pipeline/research_work.json", work)
    assert all(ref(p) == r for p, r in before.items())
    print(json.dumps(dict(preparation=ref(output), plans=plans, tests=10, runtime_tests=8, native_executed=False, admitted=False)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=("retain", "plan"))
    p.add_argument("--token", required=True)
    a = p.parse_args()
    lease(a.token, 400)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        {"retain": retain, "plan": plan}[a.operation](a.token)
    finally:
        writer.close()
