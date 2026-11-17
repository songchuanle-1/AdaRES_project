"""Close new Y98 terminal/Y99 binding preparation without replaying science."""
import argparse
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y98_deployment import Y96, Y98, git
import closure
import gpu_occupation

RUNS = (("server-1", "AdaRES-Y96", 3893460, Y96),
        ("server-2", "AdaRES-Y98", 25962, Y98))


def evidence(entry):
    return entry["path"] + " SHA" + entry["sha256"]


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y98
        before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json",
                                      "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        assert before["pipeline/goal.json"]["sha256"] == GOAL
        assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == PAUSE
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        assert running == {"server-1": {"0": "AdaRES-Y96"}, "server-2": {"0": "AdaRES-Y98"}}
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {r["id"]: r for r in work["candidates"]}
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        context = tick.load(ROOT / "pipeline/headless/.tick_context.json")
        assert not view["preflight"]["operational_issues"]
        assert view["queue"]["ready_waiting_count"] == 0
        assert view["changes"]["new_terminal_ids"] == ["AdaRES-Y95"]
        y95, y98, y99 = (candidates["AdaRES-" + n] for n in ("Y95", "Y98", "Y99"))
        old_terminal = tick.load(closure.artifact(ROOT, y95["terminal_evidence"]))
        assert old_terminal["token"] == "da9a408055d84e01b0c213c08e252046" != token
        assert y95["scientific_status"] == "fixed30_complete_frozen_review_pending"
        previous = tick.load(ROOT / "pipeline/tick_receipts/da9a408055d84e01b0c213c08e252046.json")
        assert previous["status"] == "FINALIZED" and previous["lease_release"] == "released"
        paused = next(r for r in queue["gpu_queues"]["server-2"] if r["id"] == "AdaRES-Y57")
        assert paused["status"] == "paused" and paused["dispatch_hold"]
        assert paused["resume_requires_user_authorization"] is True
        observations, occupations = {}, {}
        for name, cid, pid, commit in RUNS:
            task = next(r for r in queue["gpu_queues"][name] if r["id"] == cid)
            assert (task["status"], task["launch_verified"], task["pid"], task["code_commit"],
                    str(task["gpu"])) == ("running", True, pid, commit, "0")
            closure.admission(ROOT, task)
            entry = candidates[cid]["latest_training_observation"]
            value = tick.load(closure.artifact(ROOT, entry))
            assert value["token"] == token and value["native_progress_confirmed"] is True
            assert value["pid"] == pid and value["source_marker"] in (commit, Y98)
            fresh(value, "observed_local")
            observations[cid] = dict(evidence=entry, phase=value["after"]["latest"], pid=pid)
            proof_ref = ref("pipeline/checks/gpu_occupation_" + name + "_release_" + token + ".json")
            fresh(tick.load(closure.artifact(ROOT, proof_ref)), "observed_at")
            held = next(r for r in work["candidates"] if r.get("stage") == "blocked"
                        and r.get("blocker_kind") == "exclusive_gpu_slot" and r.get("required_server") == name)
            server = next(r for r in context["queue"]["servers"] if r["name"] == name)
            state = gpu_occupation.validated_state(ROOT, context, held, server, queue)
            assert state["busy_gpu_ids"] == state["reserved_gpu_ids"] == [0] and not state["idle_gpu_ids"]
            occupations[name] = proof_ref
        plan_ref = y98["terminal_handoff_plan"]
        assert plan_ref["path"] == "pipeline/checks/y98_terminal_preparation_" + token + ".json"
        plan = tick.load(closure.artifact(ROOT, plan_ref))
        assert plan["token"] == token and plan["CPU_guard_tests_passed"] == 29
        assert all(plan[k] is False for k in ("checkpoint_loaded", "archived", "training_stopped",
                                              "queue_changed", "remote_mutations", "full_terminal_verification_executed"))
        location = tick.load(closure.artifact(ROOT, plan["declared_location"]))
        assert location["checkpoint_directory"] == "20261110_080627"
        assert location["checkpoint_loaded"] is False and location["remote_mutations"] is False
        binding_ref = y99["additive_version_binding"]
        binding = tick.load(closure.artifact(ROOT, binding_ref))
        assert binding["token"] == token and binding["guard_tests_passed"] == 10
        assert binding["parent_commit"] == binding["working_HEAD"] == Y98
        assert binding["matching_current_owners"] == {"server-1": "AdaRES-Y96", "server-2": "AdaRES-Y98"}
        assert binding["scientific_files_identical"] and binding["native_program_bodies_identical"]
        assert binding["only_frozen_code_commit_metadata_changed"] and binding["production_HEAD_index_queue_preserved"]
        assert not binding["native_executed"] and not binding["admitted"] and not binding["scientific_revision"]
        assert y99["code_commit"] == binding["code_commit"] == "53f5b028bdf24e8b41dd934931d1fedc99843d8d"
        assert y99["stage"] == "blocked" and y99["blocker_kind"] == "exclusive_gpu_slot"
        assert not y99.get("native_smoke_result") and not y99.get("admission_receipt")
        assert binding["actual_parent_native"] == y98["native_smoke_result"]
        assert "da9a408055d84e01b0c213c08e252046" in y98["native_smoke_result"]["path"]
        attempt = work["tick_attempt"]
        assert attempt["tick_started_at"] == view["generated_at"]
        assert attempt["kind"] == "resource_check" and attempt["result"] == "blocked"
        output = ROOT / ("pipeline/checks/y99_y98_preparation_closeout_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            observations=observations, occupations=occupations, terminal_preparation=plan_ref,
            binding=binding_ref, new_scientific_candidates=0, scientific_CPU_replays=0,
            native_GPU_tasks=0, admissions=0, launches=0, archives=0, checkpoints_loaded=False,
            terminal_lifecycle_tests=29, binding_operational_tests=10, original_Y95_archive_replayed=False,
            actual_parent_native_reused=y98["native_smoke_result"],
            wording_correction="The generated operational attempt's old 'exact Y94 parent native evidence' phrase is stale; actual_parent_native, parent_commit, runner guards and current production parent are Y98. No new native or scientific test is claimed.",
            full_original_protocol_preserved=True, formal_FIRST_or_cost_executed=False,
            evidence=[ref(__file__), plan_ref, binding_ref, y95["terminal_evidence"]]
                + [r["evidence"] for r in observations.values()] + list(occupations.values())))
        memory = ROOT / "pipeline/codex_memory.md"
        backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
        assert not backup.exists()
        original = memory.read_bytes()
        backup.write_bytes(original)
        assert backup.read_bytes() == original
        old = original.decode("utf-8").splitlines()
        lines, updated = [], set()
        for line in old[:old.index("## Last Tick")]:
            if line.startswith("- Current token "):
                line = "- Current token " + token + "; snapshot " + view["generated_at"] + "; deadline " + view["deadline_at"] + "; never extend."
                updated.add("token")
            elif line.startswith("- Exact previous seven-section memory:"):
                line = "- Exact previous seven-section memory: " + evidence(ref(backup)) + "; all history/protocols/negative evidence/bans/dedicated recovery byte-preserved; never reopening authority."
                updated.add("backup")
            elif line.startswith("- Y99 full32 "):
                line = "- Y99 full32 sparse-input mask-MLP bypass keeps original CPU scientific evidence/commite5151a722778af3eecb28d0d4f6ae75e18a60e0c. NEW identical seven-file binding " + evidence(binding_ref) + "; commit" + y99["code_commit"] + ";10 operational guards passed, both native bodies unchanged, HEADY98/index/queue/runs preserved. No CPU scientific replay/native GPU/admission/launch; future runner=run_y99_bound_native_y98.py."
                updated.add("Y99")
            elif line.startswith("- AdaRES-Y96 original PID") or line.startswith("- AdaRES-Y98 original PID"):
                cid = line.split()[1]
                row = observations[cid]
                phase = row["phase"]
                name, _, pid, commit = next(r for r in RUNS if r[1] == cid)
                line = "- " + cid + " original PID" + str(pid) + "/" + name + "GPU0, " + phase["phase"] + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"]) + "; native counters/source/YAML/private runtime/power verified " + evidence(row["evidence"]) + "; commit" + commit + "."
                updated.add(cid)
            elif line.startswith("- Y98 future exact terminal plan "):
                line = "- Y98 exact original directory20261110_080627 (not launch210741); NEW29-tested terminal plan " + evidence(plan_ref) + ". Only read-only location/source/YAML/receipt/freeze binding inspected; no checkpoint loaded/archived. Future finalize_y98_fixed30.py ONLY after natural30, exact absent original process/full Train-Val1..30/stable finite last30-best-model-optimizer-scheduler and unchanged runtime/source/280W."
                updated.add("terminal")
            elif line.startswith("- Fresh canonical occupation "):
                line = "- Fresh canonical occupation " + "; ".join(evidence(r) for r in occupations.values()) + "; both authorizedGPU0 busy/reserved; collector actual tick_attempt retained. Original GPU1 frozen FIRST/cost remain held, never moved/substituted."
                updated.add("occupation")
            elif line.startswith("- Next tick first observe "):
                line = "- Next tick first observe Y96/Y98 and dispatch qualified waiting work; no Y95 terminal/Y98 native/CPU/version/binding replays. Y99 actual native requires HEADY98, matching original Y96/server1 or Y98/server2 natural30/full finite exact checkpoint/canonical archive plus fresh exclusive authorized idle/unreservedGPU0. Changed HEAD/owner needs NEW tested binding. On native PASS finish full original selected-device freeze/admission/safe launch/progress in same tick."
                updated.add("next")
            lines.append(line)
        assert updated == {"token", "backup", "Y99", "AdaRES-Y96", "AdaRES-Y98", "terminal", "occupation", "next"}
        lines += ["## Last Tick",
            "- " + token + ": begin/design checked progress/dispatch first; two original authorizedGPU0 runs progressing, no ready waiting. Snapshot Y95 terminal is already last-tick verified/archived; no replay. New Y98 exact terminal tools29 lifecycle guards and Y99 same-byte actual-nativeY98/current-owner binding10 operational guards passed, no checkpoint loading/scientific replay.",
            "- New scientificIDs0/scientificCPUreplays0/nativeGPUtasks0/admissions0/launches0/archives0/FIRST0/cost0. Goal/old receipts/data/runtime/mappings/seed1999fixed30/350W280W/foreign jobs/bans/GPU1+Y57 pauses preserved. Both fresh canonicalGPU0 busy/reserved; actual resource_check retained. Closeout " + evidence(ref(output)) + "; official finish/receipt/release authoritative."]
        assert len(lines) < 120 and sum(r.startswith("## ") for r in lines) == 7
        memory.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        phases = [cid + " " + r["phase"]["phase"] + str(r["phase"]["epoch"]) + "/30 step" + str(r["phase"]["step"]) + "/" + str(r["phase"]["count"]) for cid, r in observations.items()]
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "运行：" + "；".join(phases) + "。原PID、源码、YAML、私有运行环境、功耗及原生步数递增已核验；两张授权GPU0当前均忙且保留给原运行。",
            "新准备：Y98精确原目录20261110_080627终态工具通过29项生命周期测试；只读定位，没有加载checkpoint或执行终态归档。Y99原有七个科学文件和两份原生检查程序正文逐字保留，当前Y98源码/原Y96-Y98资源绑定通过10项操作检查。",
            "完成：本轮仅完成上述准备及真实资源检查；科学CPU重跑0、GPU原生检查0、准入0、新训练0、归档0、FIRST和成本评测0。快照中的Y95为上一轮已核验归档，不重放。",
            "文件位置：本地D:/project/AdaRES_project；代码配置在AdaRES/，操作脚本在scripts/，实验计划/证据/报告/记忆及运行收据在pipeline/；远端原始日志与权重保留在各自已验证的项目原运行路径。"
        ], decisions=[
            "Y99继续等待匹配的原Y96/server1或Y98/server2自然固定30轮完成、完整有限原checkpoint核验和规范归档，再以新鲜独占空闲且未保留的授权GPU0执行原生YAML检查；通过后同轮完成原设备完整协议冻结、准入、安全部署与步数验证。当前不挤占训练，不重跑既有CPU/绑定/原生证据。",
            "全部server2GPU1使用和Y57单独暂停继续保留；不迁移冻结后评/成本、不重启、不使用server6、不跨项目、不改调度或功耗，不干预外部任务。保留科学goal字节、所有旧收据、原数据/私有runtime/路径映射/seed1999固定30轮/禁用项/专用恢复及完整原设备协议。无新增用户处理事项。"
        ], evidence=[ref(output), ref(backup)]))
        assert all(ref(p) == entry for p, entry in before.items())
        assert git("ls-files", "--stage", "-z") == index
        return output, notes, len(lines)
    finally:
        writer.close()


def finish(token):
    lease(token, 240)
    for _, cid, _, _ in RUNS:
        command(["scripts/y98_deployment.py", "observe", "--id", cid, "--token", token, "--label", "release"])
    for name, _, _, _ in RUNS:
        command([SKILL + "/gpu_occupation.py", "AdaRES_project", "--token", token, "--server", name, "--label", "release"])
    output, notes, lines = record(token)
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
    print(json.dumps(dict(closeout=ref(output), status=receipt["status"], postflight=receipt["postflight"],
        context_committed=True, lease_release=receipt["lease_release"], memory_lines=lines,
        native_GPU_tasks=0, admissions=0, launches=0, archives=0, checkpoint_loaded=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    finish(parser.parse_args().token)
