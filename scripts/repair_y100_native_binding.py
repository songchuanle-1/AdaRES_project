"""Preserve the failed metadata draft and bind explicit current original owners."""
import argparse
import ast
import copy
from datetime import datetime
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y99_deployment_v2 import Y98, Y99, contract, git
from y100_contract import validate_cpu
from check_ef_execution import validate_frozen
from retain_y100_cpu_native import tests, OWNERS, PAUSE
import closure


def repair(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y99
        before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        work = tick.load(ROOT / "pipeline/research_work.json")
        c = next(r for r in work["candidates"] if r["id"] == "AdaRES-Y100")
        assert c["stage"] == "blocked" and not c.get("native_smoke_result") and not c.get("admission_receipt") and not c.get("native_preparation")
        validate_cpu(tick.load(closure.artifact(ROOT, c["CPU_result"])))
        version = tick.load(closure.artifact(ROOT, c["source_manifest"]))
        assert c["additive_version_binding"] == c["source_manifest"]
        assert version["working_HEAD"] == version["parent_commit"] == Y99
        assert version["matching_current_owners"] == OWNERS and version["CPU_tested_scientific_files_exact"]
        queue = monitor.load_queue(str(ROOT))
        owner_rows = []
        for name, cid, pid, commit in (("server-1", "AdaRES-Y99", 2980836, Y99), ("server-2", "AdaRES-Y98", 25962, Y98)):
            task = next(r for r in queue["gpu_queues"][name] if r["id"] == cid)
            assert (task["status"], task["launch_verified"], task["pid"], task["code_commit"]) == ("running", True, pid, commit)
            assert monitor.load_running(str(ROOT))[name]["0"] == cid
            closure.admission(ROOT, task)
            contract(commit, Y99, cid)
            owner_rows.append({k: task[k] for k in ("id", "server", "gpu", "pid", "code_commit", "config", "log", "status", "launch_verified")})
        failure = ref("pipeline/checks/y100_native_" + token + "_runtime_guards.log")
        log = closure.artifact(ROOT, failure).read_text(encoding="utf-8")
        assert "KeyError: 'earlier_Y94_preserved'" in log and "Ran 8 tests" in log and "FAILED (errors=1)" in log
        out = ROOT / ("pipeline/checks/y100_native_" + token + "_v2.json")
        binding_path = out.with_name(out.stem + "_binding.json")
        guardlog = out.with_name(out.stem + "_runtime_guards.log")
        runner = ROOT / "scripts/run_y100_native_v2.py"
        testfile = ROOT / "scripts/test_y100_runtime_plan_v2.py"
        assert not any(p.exists() for p in (out, binding_path, guardlog, runner, testfile))
        originals = [ref("scripts/run_y100_native.py"), ref("scripts/test_y100_runtime_plan.py"), failure]
        body = (ROOT / "scripts/run_y100_native.py").read_text(encoding="utf-8")
        assert body.count("earlier_Y94_preserved") == 1
        body = body.replace("earlier_Y94_preserved", "original_Y98_Y99_preserved").replace("run_y100_native.py", "run_y100_native_v2.py")
        ast.parse(body, feature_version=(3, 8))
        runner.write_text(body, encoding="utf-8", newline="\n")
        body = (ROOT / "scripts/test_y100_runtime_plan.py").read_text(encoding="utf-8")
        body = body.replace("from run_y100_native import", "from run_y100_native_v2 import")
        body = body.replace("run_y100_native.py", "run_y100_native_v2.py").replace("self.c['source_manifest']", "self.c['additive_version_binding']")
        old = "('main_index_unchanged', False))"
        assert body.count(old) == 1
        body = body.replace(old, "('main_index_unchanged', False), ('original_Y98_Y99_preserved', False))")
        ast.parse(body, feature_version=(3, 8))
        testfile.write_text(body, encoding="utf-8", newline="\n")
        binding = copy.deepcopy(version)
        binding.update(token=token, observed_at=datetime.now().astimezone().isoformat(), preserved_original_CPU_version=c["source_manifest"],
            original_Y98_Y99_preserved=True, original_owner_rows=owner_rows,
            metadata_correction="Replace stale template-only earlier_Y94_preserved with explicitly verified original_Y98_Y99_preserved. Original draft/failure/CPU evidence/seven scientific files/native program bodies remain unchanged; no scientific or GPU replay.")
        binding["evidence"] = version["evidence"] + originals + [ref(__file__)]
        tick.save(binding_path, binding)
        c["additive_version_binding"] = ref(binding_path)
        old_plans, new_plans = {}, {}
        for name, key in (("server-1", "native_smoke_plan_server1"), ("server-2", "native_smoke_plan_server2_gpu0")):
            original = c[key]
            old_plans[name] = original
            plan = tick.load(closure.artifact(ROOT, original))
            source = closure.artifact(ROOT, plan["program"]).read_text(encoding="utf-8")
            frozen = ast.literal_eval(ast.parse(source).body[0].value)
            assert validate_frozen(c, plan, frozen) == 7
            plan["execution_helper"] = ref(runner)
            plan["planned_command"] = "python -B scripts/run_y100_native_v2.py --server " + name + " --token <CURRENT_TOKEN>"
            plan["preserved_original_plan"] = original
            plan["native_program_body_unchanged"] = True
            assert validate_frozen(c, plan, frozen) == 7
            path = out.with_name(out.stem + "_" + name.replace("-", "") + "_plan.json")
            assert not path.exists()
            tick.save(path, plan)
            c[key] = ref(path)
            new_plans[name] = ref(path)
        c["native_smoke_plan"] = new_plans["server-2"]
        c["next_action"] = c["next_action"].replace("run_y100_native.py", "run_y100_native_v2.py")
        c["evidence"].extend(originals + [ref(binding_path), ref(runner), ref(testfile)] + list(old_plans.values()) + list(new_plans.values()))
        tick.save(ROOT / "pipeline/research_work.json", work)
        tests("test_y100_runtime_plan_v2.py", guardlog, 8)
        assert all(ref(p) == r for p, r in before.items()) and git("ls-files", "--stage", "-z") == index
        assert [ref(r["path"]) for r in originals] == originals
        tick.save(out, dict(complete=True, token=token, id=c["id"], code_commit=c["code_commit"],
            CPU_version=c["source_manifest"], CPU_result=c["CPU_result"], CPU_program=c["CPU_program"],
            additive_version_binding=ref(binding_path), native_plans=new_plans, original_native_plans=old_plans,
            guard_tests=c["CPU_guard_tests"], guard_tests_passed=10, runtime_guard_tests=ref(guardlog), runtime_guard_tests_passed=8,
            same_callbacks_actually_executed_on_CPU=True, native_program_bodies_unchanged=True, original_Y98_Y99_preserved=True,
            preserved_failed_metadata_guard=failure, metadata_correction_only=True, scientific_CPU_replayed=False,
            original_data_proof=__import__('y82_server1_alternative').DATA_PROOF,
            native_executed=False, admitted=False, production_sync=False, helper=ref(__file__),
            evidence=originals + [ref(runner), ref(testfile), ref(guardlog), ref(binding_path), c["CPU_guard_tests"]] + list(old_plans.values()) + list(new_plans.values())))
        c["native_preparation"] = ref(out)
        c["evidence"].extend([ref(out), ref(guardlog), ref(__file__)])
        work["tick_attempt"] = dict(tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
            candidate_id=c["id"], kind="implementation_test", result="passed",
            command="python -B scripts/run_experiment_first_probe.py y100_cpu " + c["CPU_program"]["path"] + " --token " + token + " --seconds 600",
            observation="FIRST actual CPU YAML/fullmodel/releasedbackbone/TRAIN collation, full-appearance scalar/locality/permutation/sign/no-RNG/seven-stage W2/input-gradient/causal bypass and two finite synthetic updates passed;10 actual-evidence+8 corrected runtime guards passed. Original one-key operational metadata failure preserved; no scientific replay/guard relaxation. Native GPU unexecuted, efficacy unknown.",
            evidence=[c["CPU_result"], c["source_manifest"], ref(out)])
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(preparation=ref(out), binding=ref(binding_path), tests=10, runtime_tests=8, native_executed=False, admitted=False, preserved_metadata_failure=failure)))
    finally:
        writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True)
    repair(p.parse_args().token)
