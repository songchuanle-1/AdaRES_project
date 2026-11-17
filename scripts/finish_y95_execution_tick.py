"""Close the actual Y93 archive and Y95 launch with fresh live observations."""
import argparse
import copy
from datetime import datetime
import json
import os

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from advance_y95 import prepared, make_freeze
from y95_contract import validate_result
from y95_deployment import Y93, Y94, Y95, contract, git, require_running_parent_handoff
from run_y95_bound_native_y94 import validate_terminal
import closure
import gpu_occupation


def evidence(entry):
    return entry["path"] + " SHA" + entry["sha256"]


def update_memory(token, view, candidates, observations, occupations, output):
    path = ROOT / "pipeline/codex_memory.md"
    backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
    assert not backup.exists()
    original = path.read_bytes()
    backup.write_bytes(original)
    assert backup.read_bytes() == original
    old = original.decode("utf-8").splitlines()
    lines = []
    for line in old[:old.index("## Last Tick")]:
        if line.startswith("- Current token "):
            line = ("- Current token " + token + "; snapshot " + view["generated_at"]
                    + "; deadline " + view["deadline_at"] + "; never extend.")
        elif line.startswith("- Exact previous seven-section memory:"):
            line = ("- Exact previous seven-section memory: " + evidence(ref(backup))
                    + "; all history/bans/negative evidence/dedicated recovery byte-preserved; never reopening authority.")
        elif line.startswith("- Production HEAD/native-trained identity"):
            line = ("- Production HEAD/native-trained identityY95=" + Y95
                    + "; current observer y95_deployment.py observe --id AdaRES-Y94|AdaRES-Y95 "
                    "--token CURRENT --label NEW. Never live-observe/re-finalize archived Y93 or run old observers at new HEAD.")
        elif line.startswith("- Current AdaRES-Y93 "):
            terminal = candidates["AdaRES-Y93"]["terminal_evidence"]
            value = tick.load(closure.artifact(ROOT, terminal))
            checkpoints = value["checkpoints"]
            line = ("- Y93 original PID47380/server2GPU0/280W naturally completed in exact directory20261108_082521. "
                    "Full Train/Val1..30 and stable finite exact last30/best"
                    + str(checkpoints[1]["epoch"]) + "/model/optimizer"
                    + str(checkpoints[0]["optimizer_state_count"]) + "/scheduler30-"
                    + str(checkpoints[1]["scheduler_last_epoch"]) + " verified ONCE and canonically archived. "
                    + evidence(terminal) + "; archive " + evidence(candidates["AdaRES-Y93"]["terminal_archive_binding"])
                    + "; training best " + str(checkpoints[0]["best_miou"])
                    + " is NOT formal FIRST. Original full server2GPU0/280W review now waits onY95; no restart/migration/substitution.")
        elif line.startswith("- Y93 original PID47380/server2GPU0/280W/code"):
            line += " This original run is now exactly verified/archived; its historical helper must NOT be replayed."
        elif line.startswith("- Current AdaRES-Y94 "):
            row = observations["AdaRES-Y94"]
            phase = row["phase"]
            line = ("- Current AdaRES-Y94 original PID218513/server1GPU0/350W, " + phase["phase"]
                    + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"])
                    + "; native counter/source/YAML verified " + evidence(row["evidence"])
                    + ". Original exact directory20261109_111838 (NOT launch111829). NEW actual-Y95-native "
                    "terminal binding " + evidence(candidates["AdaRES-Y94"]["terminal_handoff_plan"])
                    + ";30 lifecycle+8 additive guards; future finalize_y94_fixed30_y95.py ONLY at HEADY94/Y95. "
                    "Original helpers/freezes/receipt/source/runtime/data/mappings retained; no Y94 checkpoint loaded/stopped/archived.")
        elif line.startswith("- Fresh canonical occupation:"):
            line = ("- Fresh canonical occupation: "
                    + "; ".join(evidence(entry) for entry in occupations.values())
                    + "; both authorizedGPU0 busy/reserved; age<300s required; collector actual tick_attempt retained.")
        elif line.startswith("- Next tick first observeY93/Y94"):
            line = ("- Next tick first observeY94/Y95 using y95_deployment.py and dispatch already admitted work. "
                    "Y93 archive must not be replayed. Y96/Y98/Y99 remain CPU-prepared, GPU unexecuted/unadmitted; "
                    "their HEAD/owner-specific old plans now need NEW tested same-byte bindings at currentY95 "
                    "and currentY94/server1 orY95/server2 before execution. Preserve original CPU/version/native plans; "
                    "Y96 thenY98 thenY99, no filler or Y97 retry.")
        lines.append(line)
    candidate = candidates["AdaRES-Y95"]
    native = tick.load(closure.artifact(ROOT, candidate["native_smoke_result"]))
    row = observations["AdaRES-Y95"]
    phase = row["phase"]
    where = lines.index("## Active Hypotheses") + 1
    lines[where:where] = [
        "- Y95 FIRST actual original native server2GPU0/280W PASS " + evidence(candidate["native_smoke_result"])
        + "; losses" + str(native["losses"]) + ", peak_reserved" + str(native["peak_reserved_bytes"])
        + ", two original TRAIN finite backward/optimizer steps/four distinct scenes/full all-query mean/"
        "six self-language/norm-FFN consumers/shared heads/original hard gates observed. No scientific CPU replay/efficacy claim.",
        "- Y95 current original server2GPU0/280W PID" + str(row["pid"]) + ", " + phase["phase"]
        + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"])
        + " native counters increasing: " + evidence(row["evidence"]) + ". Exact tested source "
        + Y95 + "/seven additions only selected; full original Y93 protocol freeze "
        + evidence(candidate["post_training_freeze"]) + "; signed receipt "
        + evidence(ref(candidate["admission_receipt"])) + "; safe canonical dispatch; old receipts preserved.",
        "- Y95 original CPU/current-parent native preparation records below are immutable HISTORICAL pre-execution "
        "records, superseded only in actual execution state by the preceding evidence. Never replay CPU/version/native "
        "or old pre-admission tests that require missing native/incomplete Y93. Future HEAD/owner changes need new tested bindings.",
    ]
    lines += [
        "## Last Tick",
        "- " + token + ": natural Y93 fixed30 full exact original checkpoint/log/model/optimizer/scheduler "
        "verification and canonical archive ONCE. Original products/receipt/freezes retained, scientific benefit unmeasured.",
        "- Y95 FIRST original server2GPU0/280W native TRAIN two finite backward/optimizer steps and intended full-query "
        "bypass passed. NEW tested original Y94 additive terminal binding30+8, exact source/index-preserving selection, "
        "actual-native admission19 guards/full original protocol freeze/receipt and canonical safe dispatch completed.",
        "- Original Y94 PID218513 continues unchanged; Y95 native training counters increase. Fresh canonical bothGPU0 "
        "busy/reserved. Y96/Y98/Y99 originals and Y97 original/revision1 numerical failures preserved, no filler/relaxed gate.",
        "- New scientificIDs0/scientificCPUattempts0/nativeGPUtasks1/admissions1/launches1/archives1. No formal FIRST/cost/"
        "promotion. Exact data/runtime/mappings/goal bytes/signed old receipts/bans/dedicated recovery/350W-280W/"
        "scheduler intact. All GPU1 use and Y57 separately paused; no server6/cross-project/foreign-job changes.",
        "- Closeout " + evidence(ref(output)) + "; exact memory backup and collector actual resource_check retained. "
        "Official finish must confirm FINALIZED/PASS/context_committed/released/absent owned lease/pending archive journal.",
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
        assert git("rev-parse", "HEAD").decode().strip() == Y95
        before = {path: ref(path) for path in
                  ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        assert before["pipeline/goal.json"]["sha256"] == GOAL
        assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == PAUSE
        for cid, digest in (
            ("AdaRES-Y93", "f6b87f2813f1e8e429772bc0f63963fe1a43e0582c6c65e21771092dcbf69832"),
            ("AdaRES-Y94", "b9deb51efa3c274283f8a23ae678d42a4f0c11a29e758d19ae696e80df13cb84"),
        ):
            assert ref("pipeline/admissions/" + cid + ".json")["sha256"] == digest
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {row["id"]: row for row in work["candidates"]}
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        context = tick.load(ROOT / "pipeline/headless/.tick_context.json")
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        assert running == {"server-1": {"0": "AdaRES-Y94"}, "server-2": {"0": "AdaRES-Y95"}}
        paused = next(row for row in queue["gpu_queues"]["server-2"] if row["id"] == "AdaRES-Y57")
        assert paused["status"] == "paused" and paused["dispatch_hold"]
        assert paused["resume_requires_user_authorization"] is True
        done = monitor.load_experiment(str(ROOT), "AdaRES-Y93")
        assert done["status"] == "done_complete" and done["code_commit"] == Y93
        terminal = tick.load(closure.artifact(ROOT, done["terminal_evidence"]))
        validate_terminal(terminal, "server-2")
        assert done["terminal_evidence"] == candidates["AdaRES-Y93"]["terminal_evidence"]
        closure.artifact(ROOT, candidates["AdaRES-Y93"]["terminal_archive_binding"])
        candidate, _, _, frozen = prepared(work)
        validate_result(tick.load(closure.artifact(ROOT, candidate["native_smoke_result"])), candidate, frozen)
        require_running_parent_handoff(work, candidate)
        freeze = tick.load(closure.artifact(ROOT, candidate["post_training_freeze"]))
        parentref = candidates["AdaRES-Y93"]["post_training_freeze"]
        draft = make_freeze(tick.load(closure.artifact(ROOT, parentref)), parentref)
        assert freeze["training"] == draft["training"] and freeze["evaluation"] == draft["evaluation"]
        assert freeze["resource_condition"] == draft["resource_condition"]
        observations, occupations = {}, {}
        for name, cid, commit in (("server-1", "AdaRES-Y94", Y94), ("server-2", "AdaRES-Y95", Y95)):
            task = next(row for row in queue["gpu_queues"][name] if row["id"] == cid)
            assert task["status"] == "running" and task["launch_verified"] is True
            assert str(task["gpu"]) == "0" and task["code_commit"] == commit
            closure.admission(ROOT, task)
            contract(commit, Y95, cid)
            entry = candidates[cid]["latest_training_observation"]
            value = tick.load(closure.artifact(ROOT, entry))
            assert value["token"] == token and value["native_progress_confirmed"] is True
            assert value["pid"] == task["pid"] and value["source_marker"] in (commit, Y95)
            fresh(value, "observed_local")
            observations[cid] = dict(evidence=entry, phase=value["after"]["latest"], pid=task["pid"])
            proof_ref = ref("pipeline/checks/gpu_occupation_" + name + "_release_" + token + ".json")
            fresh(tick.load(closure.artifact(ROOT, proof_ref)), "observed_at")
            held = next(row for row in work["candidates"] if row.get("stage") == "blocked"
                and row.get("blocker_kind") == "exclusive_gpu_slot" and row.get("required_server") == name)
            server = next(row for row in context["queue"]["servers"] if row["name"] == name)
            state = gpu_occupation.validated_state(ROOT, context, held, server, queue)
            assert state["busy_gpu_ids"] == state["reserved_gpu_ids"] == [0] and not state["idle_gpu_ids"]
            occupations[name] = proof_ref
        output = ROOT / ("pipeline/checks/y95_execution_closeout_" + token + ".json")
        assert not output.exists()
        dependencies = [ref(__file__), done["terminal_evidence"],
            candidates["AdaRES-Y93"]["terminal_archive_binding"], candidate["native_smoke_result"],
            candidate["post_training_freeze"], ref(candidate["admission_receipt"]),
            candidates["AdaRES-Y94"]["terminal_handoff_plan"],
            ref("pipeline/checks/adares-y95_source_selection_" + token + ".json"),
            ref("pipeline/checks/adares-y95_dispatch_" + token + ".json")] + [
                row["evidence"] for row in observations.values()] + list(occupations.values())
        tick.save(output, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            exact_original_Y93_fixed30_verified_and_archived=True,
            actual_Y95_native_steps=2, new_scientific_candidates=0, scientific_CPU_attempts=0,
            native_GPU_tasks=1, new_admissions=1, training_launches=1, archives=1,
            original_Y94_process_unchanged=True, original_Y94_checkpoints_loaded=False,
            full_original_frozen_protocol_preserved=True, scientific_efficacy_measured=False,
            observations=observations, occupations=occupations, evidence=dependencies))
        candidate["evidence"].append(ref(output))
        assert work["tick_attempt"]["kind"] == "resource_check"
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, lines = update_memory(token, view, candidates, observations, occupations, output)
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        native = tick.load(closure.artifact(ROOT, candidate["native_smoke_result"]))
        phases = []
        for cid, row in observations.items():
            phase = row["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30，步"
                          + str(phase["step"]) + "/" + str(phase["count"]))
        tick.save(notes, dict(actions=[
            "仅AdaRES_project。核验实际owner、Disabled后台调度、正式skill/experiment_first、"
            "AGENTS/goal/SERVERS/最新GPU1暂停后begin；先查进展和尝试现有合格队列。",
            "完成：原Y93自然完成固定30轮；只读验证精确原目录20261108_082521的完整Train/Val1..30、"
            "last30/best及模型/optimizer/scheduler全部有限且稳定，按原收据和freeze完成canonical归档。"
            "保留全部原检查点，未重启，训练best不是正式FIRST或方法收益。",
            "Y95首次实际server-2 GPU0/280W原生YAML/TRAIN两步loss为" + str(native["losses"])
            + "，有限backward/optimizer更新、四不同场景和完整query-mean旁路/原norm-FFN/共享head/"
            "原hard-gate消费成立。未重做CPU科学测试，不把两步当全数据资源证明或效用评测。",
            "新建Y94兼容Y95的原终态工具和实际native绑定，30生命周期+8增量guard通过；"
            "保留原Y94 PID218513、源码/YAML/精确目录20261109_111838、完整freeze、旧收据和原工具。"
            "仅七个已测Y95新增文件精确选择，其他index和工作文件不变；19项actual-native准入guard、"
            "完整原Y93训练/评测/成本协议继承、新收据、canonical安全同步和派发完成。",
            "运行：" + "；".join(phases) + "。原生训练计数递增、精确source/YAML/receipt和"
            "350W/280W已核实，canonical物理占用两张授权GPU0均busy+reserved。",
            "本轮新科学ID0、CPU科学尝试0、GPU原生1、准入1、启动1、完整归档1；"
            "Y96/Y98/Y99原准备和Y97原始/修订1失败保留。精确备份七节记忆，保留collector真实resource_check，"
            "再report/finish，以实际收据和锁释放为准，不预签结果。"
        ], decisions=[
            "优先推进已有Y95，在本轮实际native通过后完成准入和派发，不新增填充候选或重放旧科学测试。"
            "Y93完整原server-2 GPU0/280W冻结后评/成本保留，等待匹配资源，不迁移或替换；"
            "Y96/Y98/Y99因HEAD和当前owner更新，未来执行前须新同字节测试绑定，保留全部原证据。",
            "保持科学goal字节和旧签名收据，精确原数据/私有runtime/mappings、seed1999固定30轮、"
            "完整9508表达141场景/candidate+Y36-W2-B1各两次/FIRST/场景bootstrap10000-seed1999/"
            "TRAIN1487-3325/全部有害子组和指标/原设备cost。全部GPU1使用与GPU1冻结后评继续暂停，"
            "Y57独立暂停；禁server6/跨项目/调度或功耗变化/外部任务干预。无新用户处理事项。"
        ], evidence=[ref(output), backup]))
        assert all(ref(path) == entry for path, entry in before.items())
        return output, notes, lines
    finally:
        writer.close()


def finish(token):
    lease(token, 240)
    for cid in ("AdaRES-Y94", "AdaRES-Y95"):
        command(["scripts/y95_deployment.py", "observe", "--id", cid, "--token", token, "--label", "release"])
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
    print(json.dumps(dict(closeout=ref(output), status=receipt["status"], postflight=receipt["postflight"],
        context_committed=True, lease_release=receipt["lease_release"], memory_lines=lines,
        archive_pending_files=0, new_candidates=0, native_GPU_tasks=1, admissions=1, launches=1, archives=1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    finish(parser.parse_args().token)
