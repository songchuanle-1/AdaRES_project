"""Close one real CPU preparation with fresh evidence of the two original runs."""
import argparse
import ast
import base64
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y99_deployment_v2 import Y98, Y99, git
from prepare_y100_appearance_input_skip import files
from y100_contract import validate_cpu
from run_y100_native_v2 import runtime, validate_binding, PARENT
from check_ef_execution import validate_frozen
import closure
import gpu_occupation

RUNS = (("server-1", "AdaRES-Y99", 2980836, Y99), ("server-2", "AdaRES-Y98", 25962, Y98))


def evidence(r):
    return r["path"] + " SHA" + r["sha256"]


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == PARENT == Y99
        before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        assert before["pipeline/goal.json"]["sha256"] == GOAL and ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == PAUSE
        work = tick.load(ROOT / "pipeline/research_work.json")
        c = {r["id"]: r for r in work["candidates"]}
        new, y99 = c["AdaRES-Y100"], c["AdaRES-Y99"]
        assert new["stage"] == "blocked" and new["blocker_kind"] == "exclusive_gpu_slot"
        assert new["resource_work_kind"] == "native_smoke" and not new.get("native_smoke_result") and not new.get("admission_receipt")
        cpu = tick.load(closure.artifact(ROOT, new["CPU_result"]))
        validate_cpu(cpu)
        version = tick.load(closure.artifact(ROOT, new["source_manifest"]))
        source = tick.load(closure.artifact(ROOT, version["source_snapshot"]))
        assert version["CPU_result"] == new["CPU_result"] and version["code_commit"] == new["code_commit"]
        payload = source["CPU_tested_payload"]
        assert payload == ast.literal_eval(ast.parse(closure.artifact(ROOT, new["CPU_program"]).read_text(encoding="utf-8")).body[0].value)
        for path in files():
            v = payload["variant"]["config_data"] if path.suffix == ".yaml" else payload["variant"]["python_files"][path.name]
            assert path.read_bytes() == base64.b64decode(v["base64"]) == git("show", new["code_commit"] + ":" + path.relative_to(ROOT).as_posix())
            assert ref(path)["sha256"] == v["sha256"]
        assert git("rev-parse", new["code_commit"] + "^").decode().strip() == Y99
        assert len(version["exact_delta"]) == 7 and version["main_index_unchanged"] is True
        binding = tick.load(closure.artifact(ROOT, new["additive_version_binding"]))
        validate_binding(binding, new, Y99)
        assert binding["preserved_original_CPU_version"] == new["source_manifest"]
        prep = tick.load(closure.artifact(ROOT, new["native_preparation"]))
        assert prep["token"] == token and prep["CPU_result"] == new["CPU_result"]
        assert prep["guard_tests_passed"] == 10 and prep["runtime_guard_tests_passed"] == 8
        assert prep["native_executed"] is False and prep["admitted"] is False and prep["production_sync"] is False
        assert prep["metadata_correction_only"] and prep["native_program_bodies_unchanged"] and not prep["scientific_CPU_replayed"]
        assert "Ran 10 tests" in closure.artifact(ROOT, prep["guard_tests"]).read_text(encoding="utf-8")
        assert "Ran 8 tests" in closure.artifact(ROOT, prep["runtime_guard_tests"]).read_text(encoding="utf-8")
        failure = prep["preserved_failed_metadata_guard"]
        assert "KeyError: 'earlier_Y94_preserved'" in closure.artifact(ROOT, failure).read_text(encoding="utf-8")
        servers = monitor.load_servers(str(ROOT))
        for name, key in (("server-1", "native_smoke_plan_server1"), ("server-2", "native_smoke_plan_server2_gpu0")):
            plan = tick.load(closure.artifact(ROOT, new[key]))
            program = closure.artifact(ROOT, plan["program"]).read_text(encoding="utf-8")
            frozen = ast.literal_eval(ast.parse(program).body[0].value)
            assert validate_frozen(new, plan, frozen) == 7
            assert runtime(name, new, frozen, servers[name]) == (350. if name == "server-1" else 280.)
            assert plan["execution_helper"] == ref("scripts/run_y100_native_v2.py")
            old = tick.load(closure.artifact(ROOT, prep["original_native_plans"][name]))
            assert old["program"] == plan["program"]
            assert plan["preserved_original_plan"] == prep["original_native_plans"][name]
        handoff = tick.load(closure.artifact(ROOT, y99["terminal_handoff_plan"]))
        assert handoff["token"] == token and handoff["CPU_guard_tests_passed"] == 30
        assert handoff["full_terminal_verification_executed"] is False and not handoff["checkpoint_loaded"] and not handoff["archived"]
        location = tick.load(closure.artifact(ROOT, handoff["declared_location"]))
        assert location["checkpoint_directory"] == "20261111_180638"
        assert location["checkpoint_loaded"] is False and location["remote_mutations"] is False
        assert handoff["payload"]["allowed_markers"] == [Y99]
        for d in handoff["dependencies"]:
            closure.artifact(ROOT, d)
        queue = monitor.load_queue(str(ROOT))
        assert monitor.load_running(str(ROOT)) == {"server-1": {"0": "AdaRES-Y99"}, "server-2": {"0": "AdaRES-Y98"}}
        context = tick.load(ROOT / "pipeline/headless/.tick_context.json")
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        assert not view["preflight"]["operational_issues"]
        observations, occupations = {}, {}
        for name, cid, pid, commit in RUNS:
            task = next(r for r in queue["gpu_queues"][name] if r["id"] == cid)
            assert (task["status"], task["launch_verified"], task["pid"], task["code_commit"], str(task["gpu"])) == ("running", True, pid, commit, "0")
            closure.admission(ROOT, task)
            entry = c[cid]["latest_training_observation"]
            observed = tick.load(closure.artifact(ROOT, entry))
            assert observed["token"] == token and observed["pid"] == pid and observed["native_progress_confirmed"] is True
            fresh(observed, "observed_local")
            observations[cid] = dict(evidence=entry, phase=observed["after"]["latest"], pid=pid)
            proof = ref("pipeline/checks/gpu_occupation_" + name + "_release_" + token + ".json")
            fresh(tick.load(closure.artifact(ROOT, proof)), "observed_at")
            held = new if name == "server-2" else c["AdaRES-Y96"]
            server_view = next(r for r in context["queue"]["servers"] if r["name"] == name)
            state = gpu_occupation.validated_state(ROOT, context, held, server_view, queue)
            assert state["busy_gpu_ids"] == state["reserved_gpu_ids"] == [0] and not state["idle_gpu_ids"]
            occupations[name] = proof
        pause = next(r for r in queue["gpu_queues"]["server-2"] if r["id"] == "AdaRES-Y57")
        assert pause["status"] == "paused" and pause["dispatch_hold"] and pause["resume_requires_user_authorization"]
        out = ROOT / ("pipeline/checks/y100_preparation_closeout_" + token + ".json")
        assert not out.exists()
        tick.save(out, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            Y100_CPU=new["CPU_result"], Y100_source=new["source_manifest"], Y100_native_preparation=new["native_preparation"],
            Y99_terminal_preparation=y99["terminal_handoff_plan"], observations=observations, occupations=occupations,
            metadata_failure_preserved=failure, CPU_scientific_attempts=1, scientific_CPU_replays=0,
            new_scientific_candidates=1, native_GPU_tasks=0, admissions=0, launches=0, archives=0,
            full_checkpoint_verification_executed=False, formal_FIRST_or_cost_executed=False,
            evidence=[ref(__file__), new["CPU_result"], new["CPU_program"], new["source_manifest"], new["additive_version_binding"],
                new["native_preparation"], y99["terminal_handoff_plan"], failure] + [r["evidence"] for r in observations.values()] + list(occupations.values())))
        path = ROOT / "pipeline/codex_memory.md"
        backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
        assert not backup.exists()
        old_bytes = path.read_bytes()
        backup.write_bytes(old_bytes)
        assert backup.read_bytes() == old_bytes
        old = old_bytes.decode("utf-8").splitlines()
        lines, updated = [], set()
        for line in old[:old.index("## Last Tick")]:
            if line.startswith("- Current token "):
                line = "- Current token " + token + "; snapshot " + view["generated_at"] + "; deadline " + view["deadline_at"] + "; never extend."
                updated.add("token")
            elif line.startswith("- Exact previous seven-section memory:"):
                line = "- Exact previous seven-section memory: " + evidence(ref(backup)) + "; complete historical routes/protocols/negative evidence/bans/dedicated recovery byte-preserved; never reopening authority."
                updated.add("backup")
            elif line.startswith("- AdaRES-Y98 original PID") or line.startswith("- AdaRES-Y99 original PID"):
                cid = "AdaRES-Y98" if line.startswith("- AdaRES-Y98") else "AdaRES-Y99"
                row = observations[cid]
                p = row["phase"]
                server = "server2GPU0/280W" if cid.endswith("Y98") else "server1GPU0/350W"
                line = "- " + cid + " original PID" + str(row["pid"]) + "/" + server + ", " + p["phase"] + str(p["epoch"]) + "/30 step" + str(p["step"]) + "/" + str(p["count"]) + "; original source/YAML/private runtime/power/native counter increase verified " + evidence(row["evidence"]) + "; original receipt/launch/freeze unchanged."
                updated.add(cid)
            elif line.startswith("- Fresh canonical occupation "):
                line = "- Fresh canonical occupation " + "; ".join(evidence(r) for r in occupations.values()) + "; both authorizedGPU0 busy/reserved by originalY98/Y99. Y100 native waits; original GPU1 full frozen FIRST/cost remain user-held and never moved/substituted."
                updated.add("occupation")
            elif line.startswith("- Next tick observe "):
                line = "- Next tick first observe Y98/Y99, dispatch qualified waiting work. Y98 ONLY finalize_y98_fixed30_y99_v2.py after original natural30 and full exact terminal verification. Y100 FIRST native run_y100_native_v2.py ONLY at HEADY99 after matching original Y98/server2 or Y99/server1 natural30/full finite exact checkpoint/canonical archive plus fresh exclusive authorized idle/unreservedGPU0; then full original selected-device freeze/admission/source selection/dispatch/increasing counters same tick. No CPU/version/plan or historical terminal replays."
                updated.add("next")
            lines.append(line)
        assert updated == {"token", "backup", "AdaRES-Y98", "AdaRES-Y99", "occupation", "next"}
        at = lines.index("## Closed Or Failed Directions")
        lines[at:at] = [
            "- Y100 independent W2 full1024 raw-appearance bypass beside complete original input_proj_2d BEFORE original sum_norm/visual memory; +262144/total164891781/no RNG; complete static mask/readouts/original projections/full inputs/shared confidence/refinement/SCG/hard gates/data/seed1999/fixed30 unchanged. Not Y98 appearance-to-mask/Y99 spatial-to-mask/Y71 projected-product/Y97 language-square; no combination/reopening/tuning.",
            "- Y100 FIRST original CPU YAML/fullmodel/releasedbackbone/TRAIN collation PASS " + evidence(new["CPU_result"]) + ";162738points/8expressions, scalarfloat32-64/permutation/sign/row-local/noRNG/seven-stage W2 anchors/input gradients/causal initial-final masks/exact bypass/two finite synthetic updates; original projection/sum_norm/first visual memory/static7/shared heads/gates/14 original visual parameter gradients verified. CPU is NOT native/efficacy.",
            "- Y100 exact seven tested additions retained isolated branch " + new["retained_branch"] + " commit" + new["code_commit"] + "; " + evidence(new["source_manifest"]) + "; parentHEADY99/mainindex/current production preserved.10 actual-evidence +8 corrected runtime guards PASS " + evidence(new["native_preparation"]) + "; current original owners binding " + evidence(new["additive_version_binding"]) + "; native runner=run_y100_native_v2.py/server1GPU0-350W or server2GPU0-280W. GPU native/admission/launch remain unexecuted.",
            "- Y100 initial8-runtime-guard draft had ONE metadata KeyError earlier_Y94_preserved; exact failure " + evidence(failure) + "; originals preserved, v2 explicitly validates original_Y98_Y99_preserved against exact owners/receipts/source. Both frozen native program bodies/seven scientific files/CPU result unchanged; no scientific retry/tolerance relaxation. Never use first runner/plans."]
        at = lines.index("## Open Questions And Risks")
        lines[at:at] = ["- Y99 exact original directory20261111_180638 discovered read-only (not inferred launch timestamp),30 lifecycle guards PASS " + evidence(y99["terminal_handoff_plan"]) + "; future finalize_y99_fixed30.py ONLY at HEADY99 after natural30/full exact Train-Val1..30/absent original PID/stable finite last-best-model-optimizer-scheduler and original runtime/source/350W. No live checkpoint loaded or archive executed."]
        lines += ["## Last Tick", "- " + token + ": observed/dispatched existing qualified work first; no new physical terminal/operational issue, previous Y96 archive retained without replay. Original Y98/Y99 increasing counters, Y99 exact original terminal30 checks, Y100 FIRST CPU feasibility/isolated version/10+8 operational guards/native plans completed; both authorizedGPU0 fresh busy/reserved.",
            "- New scientificIDs1/CPUscientificattempts1/CPUreplays0/nativeGPUtasks0/admissions0/launches0/archives0/FIRST0/cost0. Goal/original receipts/data/runtime/mappings/350W280W/fixed30/bans/GPU1+Y57 pauses/foreign jobs preserved. Closeout " + evidence(ref(out)) + "; official finish/receipt/release authoritative."]
        assert len(lines) < 120 and sum(line.startswith("## ") for line in lines) == 7
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        phases = [cid + " " + r["phase"]["phase"] + str(r["phase"]["epoch"]) + "/30 step" + str(r["phase"]["step"]) + "/" + str(r["phase"]["count"]) for cid, r in observations.items()]
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "运行：" + "；".join(phases) + "。两张授权GPU0均按原进程/源码/原生YAML/私有环境/350W280W训练，步数递增。无新物理终态，保留上轮Y96完整核验归档，不重放。",
            "完成准备：Y99实际原目录20261111_180638只读发现并绑定，30项终态保护检查通过；未加载活跃权重、未归档、不停止训练。",
            "新准备：Y100独立W2完整1024通道外观绕过原外观MLP到原sum_norm/视觉记忆，保留完整投影和全部输入。首次CPU原生YAML/完整模型/发布backbone/TRAIN加载通过，162738点8表达；标量float32/64、置换/局部性/符号/无RNG、七阶段基线与输入梯度对齐、因果旁路恢复及两次有限synthetic decoder更新通过。CPU结果不是GPU原生检查或效果证明。",
            "保存七个实际测试科学文件到隔离版本，生产HEAD/主索引/原运行不变。10项实际证据和修正后8项运行计划检查通过；原单项旧字段元数据失败日志及原稿保留。两套原生GPU程序体不变，无科学重跑或放宽控制。"
        ], decisions=[
            "两张授权GPU0均实测busy/reserved，Y100原生检查等待匹配原Y98或Y99自然固定30轮、完整精确有限检查点核验和规范归档，再验空闲独占资源。原生检查通过后同轮冻结完整原设备后评/成本协议、准入并尝试派发；当前不入队、不声称GPU PASS或效果。",
            "仅AdaRES；server2GPU1全部使用与Y57单独暂停保持，不迁移冻结后评/成本、不重启受保护实验、不使用server6、不改数据/功耗/调度、不干预外部任务。新科学ID1/CPU科学尝试1/GPU原生0/准入0/训练0/归档0/FIRST0/成本0；无新增用户处理事项。"
        ], evidence=[ref(out), ref(backup)]))
        assert all(ref(p) == r for p, r in before.items()) and git("ls-files", "--stage", "-z") == index
        return out, notes, len(lines)
    finally:
        writer.close()


def finish(token):
    lease(token, 240)
    for _, cid, _, _ in RUNS:
        command(["scripts/y99_deployment_v2.py", "observe", "--id", cid, "--token", token, "--label", "release"])
    for name, _, _, _ in RUNS:
        command([SKILL + "/gpu_occupation.py", "AdaRES_project", "--token", token, "--server", name, "--label", "release"])
    out, notes, lines = record(token)
    command([SKILL + "/foreground_tick.py", "AdaRES_project", "report", "--token", token, "--notes", str(notes)])
    command([SKILL + "/foreground_tick.py", "AdaRES_project", "finish", "--token", token])
    receipt = tick.load(ROOT / ("pipeline/tick_receipts/" + token + ".json"))
    assert receipt["status"] == "FINALIZED" and receipt["postflight"] == "PASS"
    assert receipt["context_committed"] is True and receipt["lease_release"] == "released"
    assert not (ROOT / "pipeline/headless/.foreground.lockdir").exists()
    assert not (ROOT / "pipeline/headless/.tick_context_state.next.json").exists()
    pending = ROOT / "pipeline/archive_pending"
    assert not pending.exists() or not any(p.is_file() for p in pending.rglob("*"))
    assert ref("pipeline/goal.json")["sha256"] == GOAL
    print(json.dumps(dict(closeout=ref(out), status=receipt["status"], postflight=receipt["postflight"], context_committed=True,
        lease_release=receipt["lease_release"], memory_lines=lines, observations=tick.load(out)["observations"],
        new_scientific_candidates=1, native_GPU_tasks=0, admissions=0, launches=0, archives=0)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True)
    finish(p.parse_args().token)
