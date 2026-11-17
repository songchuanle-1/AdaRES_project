"""Close tested Y95/Y94 operational preparation without inventing GPU results."""
import argparse
import copy
from datetime import datetime
import json
import os

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y94_deployment import Y93, Y94, contract, git
from rebind_y95_on_y94 import OLD, OLD_BRANCH, BRANCH, RUNNER, OWNERS, compare_programs
from check_ef_execution import validate_frozen
from y95_contract import validate_cpu
import closure
import gpu_occupation


def evidence(entry):
    return entry["path"] + " SHA" + entry["sha256"]


def save_memory(token, view, candidates, observations, occupations, output):
    path = ROOT / "pipeline/codex_memory.md"
    backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
    assert not backup.exists()
    original = path.read_bytes()
    backup.write_bytes(original)
    assert backup.read_bytes() == original
    old = original.decode("utf-8").splitlines()
    assert sum(line.startswith("## ") for line in old) == 7
    y94, y95 = candidates["AdaRES-Y94"], candidates["AdaRES-Y95"]
    lines = []
    for line in old[:old.index("## Last Tick")]:
        if line.startswith("- Current token "):
            line = ("- Current token " + token + "; snapshot " + view["generated_at"]
                    + "; deadline " + view["deadline_at"] + "; never extend.")
        elif line.startswith("- Exact previous seven-section memory:"):
            line = ("- Exact previous seven-section memory: " + evidence(ref(backup))
                    + "; ALL prior history/negative evidence/bans/dedicated restoration requirements "
                    "byte-preserved in the complete snapshot chain, never permission to reopen.")
        elif line.startswith("- Y95 retained seven tested A files"):
            line = ("- Y95 CURRENT identical seven CPU-tested additions retained on actual-nativeY94: "
                    "commit" + y95["code_commit"] + "/" + BRANCH + "; binding "
                    + evidence(y95["additive_version_binding"])
                    + ",10 NEW operational guards passed; original commit" + OLD + "/" + OLD_BRANCH
                    + " and complete original binding/native plans preserved in original_pre_y94_binding. "
                    "No production HEAD/index/queue/source or scientific CPU program/body changes.")
        elif line.startswith("- Y95 runner="):
            line = ("- Y95 ONLY current runner=" + RUNNER + " at HEADY94 with current owners "
                    "Y94/server1 orY93/server2 natural30/full finite exact original checkpoint/"
                    "canonical archive plus fresh authorized exclusive idle/unreservedGPU0. BOTH "
                    "native program bodies and frozen scientific/runtime payloads unchanged; only "
                    "code_commit metadata changed. Native GPU/result/admission still UNEXECUTED. "
                    "Further HEAD/owner change requires NEW tested binding; no CPU/version/plan replay.")
        elif line.startswith("- Current AdaRES-Y93 "):
            entry = observations["AdaRES-Y93"]
            phase = entry["phase"]
            line = ("- Current AdaRES-Y93 " + phase["phase"] + str(phase["epoch"])
                    + "/30 step" + str(phase["step"]) + "/" + str(phase["count"])
                    + "; original PID47380/source/YAML/native progress/280W verified: "
                    + evidence(entry["evidence"]))
        elif line.startswith("- Current AdaRES-Y94 "):
            entry = observations["AdaRES-Y94"]
            phase = entry["phase"]
            line = ("- Current AdaRES-Y94 original PID218513/server1GPU0/350W, "
                    + phase["phase"] + str(phase["epoch"]) + "/30 step" + str(phase["step"])
                    + "/" + str(phase["count"]) + "; native counter increase/source/YAML verified "
                    + evidence(entry["evidence"]) + ". Actual original save directory20261109_111838 "
                    "from native log declaration, NOT launch111829. NEW30-guard terminal plan "
                    + evidence(y94["terminal_handoff_plan"])
                    + "; future helperfinalize_y94_fixed30.py ONLY at HEADY94, preserves original "
                    "source/YAML/admission/freeze/private runtime/exact data/mappings. No checkpoint "
                    "loaded/stopped/archived; changed source requires NEW tested additive binding.")
        elif line.startswith("- Fresh canonical occupation:"):
            line = ("- Fresh canonical occupation: "
                    + "; ".join(evidence(entry) for entry in occupations.values())
                    + "; both authorizedGPU0 busy/reserved, age<300s required; collector actual tick_attempt retained.")
        elif line.startswith("- Next tick first observeY93/Y94"):
            line = ("- Next tick first observeY93/Y94 and dispatch already admitted work. Y95 "
                    "same-byte HEADY94/current-owner binding is ready; native GPU still awaits matching "
                    "natural30/full exact finite checkpoint/archive and fresh idleGPU0, then actual native "
                    "PASS/full original freeze/receipt/admission/source selection/safe dispatch/increasing "
                    "counters in that tick. Y96/Y98 still need NEW tested same-byte bindings on currentY94. "
                    "No obsolete HEADY93 runner/CPU/version/terminal replay or Y97 numerical-hold retry.")
        lines.append(line)
    lines += [
        "## Last Tick",
        "- " + token + ": actual original Y93/Y94 native counters/source/YAML/receipt/private runtimes "
        "and280W/350W verified. Preflight new_terminalY92 describes the PRIOR tick's archive, "
        "not another completion/archive; never repeat its terminal helper or checkpoint reads.",
        "- Earlier Y95 SAME scientific bytes retained additively on actual-nativeY94;10 new operational "
        "guards pass, original scientific CPU result/program/native callbacks and both runtime bodies "
        "preserved. HEAD/index/queue and Y93/Y94 original runs untouched; no native-result fixture/PASS.",
        "- Prepared exact Y94 lifecycle/source/YAML/admission/full-freeze/private-runtime/location "
        "contract; actual native log directory20261109_111838 confirmed read-only.30 new lifecycle "
        "guards pass; no checkpoint loaded, native GPU test, stop or archive. Y93 prior29+8 chain unchanged.",
        "- New scientificIDs0/scientificCPUattempts0/nativeGPUtasks0/admissions0/launches0/archives0; "
        "new operational groups2(10+30 guards). Y96/Y98 originals and Y97 original/revision1 failures "
        "frozen; no extra revision, tolerance relaxation, renamed candidate or user permission request.",
        "- Original goal/receipts/full9508x141 FIRST/two repeats/bootstrap10000 seed1999/TRAIN1487-3325/"
        "all harm metrics/original-device cost/data/private runtimes/mappings/fixed30/bans/dedicated restore "
        "rules/power/scheduler intact. GPU1 all use and Y57 separately paused; no server6/cross-project/foreign-job change.",
        "- Closeout " + evidence(ref(output))
        + "; exact memory backup, fresh sequential native observations and canonical physical occupation, "
        "collector resource_check preserved; official report/finish must verify "
        "FINALIZED/PASS/context_committed/released/absent owned lock and pending archive journal.",
    ]
    assert len(lines) < 120 and sum(line.startswith("## ") for line in lines) == 7
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return ref(backup), len(lines)


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert os.environ["CODEX_THREAD_ID"] == "01a07e32-78bf-7670-925b-370911ea48db"
        assert git("rev-parse", "HEAD").decode().strip() == Y94
        before = {path: ref(path) for path in
                  ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        assert before["pipeline/goal.json"]["sha256"] == GOAL
        assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == PAUSE
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {row["id"]: row for row in work["candidates"]}
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        context = tick.load(ROOT / "pipeline/headless/.tick_context.json")
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        assert running == {"server-1": {"0": "AdaRES-Y94"}, "server-2": {"0": "AdaRES-Y93"}}
        servers = monitor.load_servers(str(ROOT))
        assert servers["server-1"]["allowed_gpus"] == servers["server-2"]["allowed_gpus"] == [0]
        assert servers["server-6"]["allowed_gpus"] == []
        paused = next(row for row in queue["gpu_queues"]["server-2"] if row["id"] == "AdaRES-Y57")
        assert paused["status"] == "paused" and paused["dispatch_hold"]
        assert paused["resume_requires_user_authorization"] is True
        observations, occupations = {}, {}
        for name, cid, pid, commit in (
            ("server-1", "AdaRES-Y94", 218513, Y94), ("server-2", "AdaRES-Y93", 47380, Y93)
        ):
            task = next(row for row in queue["gpu_queues"][name] if row["id"] == cid)
            assert (task["status"], task["launch_verified"], task["pid"], task["code_commit"],
                    str(task["gpu"])) == ("running", True, pid, commit, "0")
            closure.admission(ROOT, task)
            contract(commit, Y94, cid)
            entry = candidates[cid]["latest_training_observation"]
            value = tick.load(closure.artifact(ROOT, entry))
            assert value["token"] == token and value["native_progress_confirmed"] is True
            assert value["pid"] == pid and value["source_marker"] in (commit, Y94)
            fresh(value, "observed_local")
            observations[cid] = dict(evidence=entry, phase=value["after"]["latest"], pid=pid)
            proof_ref = ref("pipeline/checks/gpu_occupation_" + name + "_release_" + token + ".json")
            proof = tick.load(closure.artifact(ROOT, proof_ref))
            fresh(proof, "observed_at")
            held = next(row for row in work["candidates"] if row.get("stage") == "blocked"
                and row.get("blocker_kind") == "exclusive_gpu_slot" and row.get("required_server") == name)
            snapshot_server = next(row for row in context["queue"]["servers"] if row["name"] == name)
            state = gpu_occupation.validated_state(ROOT, context, held, snapshot_server, queue)
            assert state["busy_gpu_ids"] == state["reserved_gpu_ids"] == [0] and not state["idle_gpu_ids"]
            occupations[name] = proof_ref
        y93, y94, y95 = (candidates["AdaRES-" + name] for name in ("Y93", "Y94", "Y95"))
        assert y95["stage"] == "blocked" and y95["blocker_kind"] == "exclusive_gpu_slot"
        assert not y95.get("native_smoke_result") and not y95.get("admission_receipt")
        assert not (ROOT / "pipeline/admissions/AdaRES-Y95.json").exists()
        binding_ref = y95["additive_version_binding"]
        binding = tick.load(closure.artifact(ROOT, binding_ref))
        assert binding["token"] == token and binding["parent_commit"] == binding["working_HEAD"] == Y94
        assert binding["guard_tests_passed"] == 10 and binding["native_executed"] is False
        assert binding["admitted"] is False and binding["scientific_revision"] is False
        assert binding["matching_current_owners"] == OWNERS
        assert binding["scientific_files_identical"] and binding["native_program_bodies_identical"]
        assert binding["only_frozen_code_commit_metadata_changed"] and binding["production_HEAD_index_queue_preserved"]
        for entry in binding["evidence"]:
            closure.artifact(ROOT, entry)
        assert git("rev-parse", OLD_BRANCH).decode().strip() == OLD
        assert git("rev-parse", BRANCH).decode().strip() == y95["code_commit"]
        validate_cpu(tick.load(closure.artifact(ROOT, y95["CPU_result"])))
        for name, key in (("server-1", "native_smoke_plan_server1"), ("server-2", "native_smoke_plan_server2_gpu0")):
            old = tick.load(closure.artifact(ROOT, y95["original_pre_y94_binding"][key]))
            new = tick.load(closure.artifact(ROOT, y95[key]))
            frozen = compare_programs(old["program"], new["program"], y95, OLD)
            assert validate_frozen(y95, new, frozen) == 7
            assert new["execution_helper"] == ref(RUNNER)
        terminal_ref = y94["terminal_handoff_plan"]
        terminal = tick.load(closure.artifact(ROOT, terminal_ref))
        assert terminal["token"] == token and terminal["CPU_guard_tests_passed"] == 30
        assert terminal["admission"] == ref("pipeline/admissions/AdaRES-Y94.json")
        assert terminal["post_training_freeze"] == y94["post_training_freeze"]
        assert terminal["payload"]["allowed_markers"] == [Y94]
        assert terminal["payload"]["pid"] == 218513
        assert terminal["checkpoint_loaded"] is False and terminal["archived"] is False
        assert terminal["training_stopped"] is False and terminal["remote_mutations"] is False
        location = tick.load(closure.artifact(ROOT, terminal["declared_location"]))
        assert location["checkpoint_directory"] == "20261109_111838"
        assert location["checkpoint_loaded"] is False and location["remote_mutations"] is False
        for entry in terminal["dependencies"]:
            closure.artifact(ROOT, entry)
        for entry, count in ((ref("pipeline/checks/y95_y94_additive_binding_" + token + ".log"), 10),
                             (ref("pipeline/checks/y94_terminal_preparation_" + token + "_tests.log"), 30)):
            raw = closure.artifact(ROOT, entry).read_bytes()
            assert ("Ran " + str(count) + " tests").encode() in raw and b"OK" in raw and b"skipped" not in raw
        original93 = tick.load(closure.artifact(ROOT, y93["terminal_handoff_plan"]))
        assert original93["CPU_guard_tests_passed"] == 29 and original93["additive_guard_tests_passed"] == 8
        assert original93["checkpoint_loaded"] is False and not y93.get("terminal_evidence")
        untouched = {}
        for cid in ("AdaRES-Y96", "AdaRES-Y98"):
            row = candidates[cid]
            assert row["stage"] == "blocked" and row["blocker_kind"] == "exclusive_gpu_slot"
            assert not row.get("native_smoke_result") and not row.get("admission_receipt")
            untouched[cid] = row["source_manifest"]
            for key in ("CPU_result", "source_manifest", "native_preparation",
                        "native_smoke_plan_server1", "native_smoke_plan_server2_gpu0"):
                closure.artifact(ROOT, row[key])
        numerical = candidates["AdaRES-Y97"]
        assert numerical["stage"] == "blocked" and numerical["blocker_kind"] == "numerical_control_failure"
        assert len(numerical["revision_history"]) == 1 and numerical["revision_history"][0]["result"] == "failed"
        closure.validate_revisions(ROOT, numerical)
        output = ROOT / ("pipeline/checks/y95_y94_binding_closeout_" + token + ".json")
        assert not output.exists()
        dependencies = [ref(__file__), binding_ref, terminal_ref, y93["terminal_handoff_plan"],
            numerical["CPU_failure"], numerical["revision1_result"]] + [
                entry["evidence"] for entry in observations.values()] + list(occupations.values())
        tick.save(output, dict(
            complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            Y95_same_scientific_bytes_tested_on_Y94=True, Y95_new_operational_guards=10,
            Y94_original_declared_directory="20261109_111838", Y94_new_lifecycle_guards=30,
            original_Y93_Y94_PID_training_source_yaml_receipts_full_freezes_preserved=True,
            Y94_terminal_checkpoint_loaded=False, Y94_terminal_archived=False,
            scientific_CPU_programs_or_completed_tests_replayed=False,
            Y96_Y98_unchanged=untouched, Y97_original_and_revision1_failures_preserved=True,
            new_scientific_candidates=0, scientific_CPU_candidate_attempts=0, native_GPU_tasks=0,
            new_admissions=0, new_training_launches=0, exact_original_archives=0,
            production_HEAD_index_queue_preserved=True, full_original_FIRST_and_cost_preserved=True,
            original_GPU1_and_Y57_separate_pause_preserved=True,
            observations=observations, occupations=occupations, efficacy_measured=False, evidence=dependencies))
        attempt = copy.deepcopy(work["tick_attempt"])
        assert attempt["kind"] == "resource_check"
        y95["evidence"].append(ref(output))
        y94["evidence"].append(ref(output))
        y94["next_action"] = (
            "Preserve original PID218513/server1 GPU0/350W native fixed30; observe using y94_deployment.py. "
            "Exact original native save directory20261109_111838 and30 lifecycle guards now prepared; "
            "ONLY after natural30/process absence use finalize_y94_fixed30.py at HEADY94 for complete "
            "original finite last/best/model/optimizer/scheduler verification and canonical archive. "
            "Never load live checkpoints or infer directory from launch time; changed source requires "
            "a NEW tested additive terminal binding. Preserve Y93 original run, GPU1/Y57 separate pauses. "
            "Y95 new same-byte native runner ready but GPU execution awaits matching natural30/archive/"
            "fresh exclusive idleGPU0; no restart, migration or training-metric scientific promotion.")
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, lines = save_memory(token, view, candidates, observations, occupations, output)
        decision = monitor.log_decision(str(ROOT), "other",
            "Y93/Y94持续训练且GPU0无空槽时，如何推进较早Y95并完善Y94终态安全？",
            "Y95同字节科学文件保留到实际native通过的Y94父版本，新增10项执行绑定测试；"
            "只读确认Y94原日志保存目录，新增30项精确生命周期测试，保留正在运行的两项实验。",
            "实际native正文/科学CPU程序不变、不重放；只改冻结commit元数据，原分支/计划/"
            "负结果保留。终态准备不是完整checkpoint核验或归档；无GPU原生/准入/新启动。"
            "对应GPU0须自然终态完整核验归档后实时空闲，GPU1及Y57独立暂停。",
            output.relative_to(ROOT).as_posix())
        phases = []
        for cid, entry in observations.items():
            phase = entry["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30，步"
                          + str(phase["step"]) + "/" + str(phase["count"]))
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅AdaRES_project。实际会话归属、后台Disabled、正式skill/experiment_first、"
            "AGENTS/goal/SERVERS/最新GPU1暂停核验；本轮先查进展及派发现有合格队列，"
            "ready waiting为0。preflight中的Y92新终态来自上一轮已经完成的归档，不重复终态核验或归档。",
            "运行：" + "；".join(phases) + "。两项原PID/source/YAML/receipt/私有runtime/"
            "精确数据与350W/280W及原生递增步数确认；canonical只读占用均busy+reserved，"
            "不抢占当前运行或GPU1外部任务。",
            "新准备：较早Y95的同字节七文件保留在当前实际native通过的Y94父版本，10项"
            "新操作测试实际通过。原CPU结果/程序/native callbacks与双runtime原生程序正文不变，"
            "仅冻结code_commit元数据变更；旧分支、旧计划和全部证据保留，主HEAD/index/queue不变。"
            "匹配当前owner现为Y94/server1、Y93/server2，旧HEAD/owner/GPU1/错误runtime/源/goal等均拒绝。",
            "新准备：只读核对Y94原日志声明，实际目录20261109_111838，不用启动时间111829推断。"
            "建立原PID/source/YAML/收据/完整冻结协议/私有runtime/原数据映射绑定和30项生命周期"
            "测试，首次通过；没有加载live checkpoint、停止训练、执行终态完整checkpoint核验或归档。",
            "本轮新科学ID0、科学CPU尝试0、GPU原生0、准入0、训练启动0、归档0；新增操作"
            "测试组2共10+30项。Y96/Y98原工件保持，仍需当前HEAD新绑定；Y97原始及修订1数值"
            "失败和不变控制保持，不因计时器重跑或放宽。Y93既有29+8终态兼容链原样保留。",
            "精确备份七节记忆，刷新训练及canonical物理占用并保留collector实际resource_check，"
            "再正式report/finish，不预称收尾结果。完整9508表达/141场景/FIRST主报告/各两次/"
            "场景bootstrap10000-seed1999/TRAIN1487-3325/全部有害子组与指标/原设备cost协议未改变。"
        ], decisions=[
            "优先推进已有较早Y95，不为凑额度新增或复制机制。GPU原生未执行不能冒充"
            "PASS、waiting或准入。当前HEADY94下匹配Y93/server2或Y94/server1自然30轮、"
            "完整原检查点有限性及canonical归档、实时独占空闲GPU0后运行新run_y95_bound_native_y94.py；"
            "实际通过再当轮完整freeze/receipt/准入/派发/递增counter。再变HEAD/owner须新测试绑定。",
            "Y94未来精确终态仅在自然30轮/进程消失后使用新finalize_y94_fixed30.py；"
            "Y93继续使用既有finalize_y93_fixed30_y94.py。不得加载live checkpoint，"
            "不得把日志结束、stat或本轮生命周期准备当完整checkpoint核验。历史Y81/Y82/Y83/Y92不重跑。",
            "科学goal字节、原收据、原数据/私有runtime/mappings、固定30轮seed1999、"
            "禁用项、专用恢复、功耗和调度不变。GPU1全部使用及原GPU1冻结后评继续挂起，"
            "Y57单独暂停；不迁移后评、不替代资源、不碰server6/其他项目/外部任务。无需新用户操作。"
        ], evidence=[ref(output), backup, ref(decision)]))
        assert work["tick_attempt"] == attempt
        assert all(ref(path) == value for path, value in before.items())
        assert git("ls-files", "--stage", "-z") == index
        return output, notes, lines
    finally:
        writer.close()


def finish(token):
    lease(token, 240)
    for cid in ("AdaRES-Y94", "AdaRES-Y93"):
        command(["scripts/y94_deployment.py", "observe", "--id", cid, "--token", token, "--label", "release"])
    for name in ("server-1", "server-2"):
        command([SKILL + "/gpu_occupation.py", "AdaRES_project", "--token", token,
                 "--server", name, "--label", "release"])
    output, notes, lines = record(token)
    command([SKILL + "/foreground_tick.py", "AdaRES_project", "report",
             "--token", token, "--notes", str(notes)])
    command([SKILL + "/foreground_tick.py", "AdaRES_project", "finish", "--token", token])
    receipt = tick.load(ROOT / ("pipeline/tick_receipts/" + token + ".json"))
    assert receipt["status"] == "FINALIZED" and receipt["postflight"] == "PASS"
    assert receipt["context_committed"] is True and receipt["lease_release"] == "released"
    assert not (ROOT / "pipeline/headless/.foreground.lockdir").exists()
    assert not (ROOT / "pipeline/headless/.tick_context_state.next.json").exists()
    pending = ROOT / "pipeline/archive_pending"
    assert not pending.exists() or not any(path.is_file() for path in pending.rglob("*"))
    assert ref("pipeline/goal.json")["sha256"] == GOAL
    print(json.dumps(dict(closeout=ref(output), observations=tick.load(output)["observations"],
        status=receipt["status"], postflight=receipt["postflight"], context_committed=True,
        lease_release=receipt["lease_release"], memory_lines=lines, archive_pending_files=0,
        new_admissions=0, launches=0, archives=0), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    finish(parser.parse_args().token)
