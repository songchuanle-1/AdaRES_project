"""Close the owned Y92 archive / Y94 native launch tick with fresh evidence."""
import argparse
import copy
from datetime import datetime
import json
import os

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from advance_y94 import make_freeze, prepared, validate_result, validate_terminal
from y94_deployment import Y92, Y93, Y94, contract, git
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
    y92, y93, y94 = (candidates["AdaRES-" + name] for name in ("Y92", "Y93", "Y94"))
    lines = []
    for line in old[:old.index("## Last Tick")]:
        if line.startswith("- Current token "):
            line = ("- Current token " + token + "; snapshot " + view["generated_at"]
                    + "; deadline " + view["deadline_at"] + "; never extend.")
        elif line.startswith("- Exact previous seven-section memory:"):
            line = ("- Exact previous seven-section memory: " + evidence(ref(backup))
                    + "; complete historical snapshot chain/bans/negative evidence/"
                    "dedicated restore requirements byte-preserved. Compression never reopens work.")
        elif line.startswith("- Y94 CURRENT"):
            line = ("- Y94 retained seven tested additions selected EXACTLY on Y93: commit" + Y94
                    + "; source selection "
                    + evidence(ref("pipeline/checks/adares-y94_source_selection_" + token + ".json"))
                    + "; unrelated index/working files preserved; only server1 was safely synchronized.")
        elif line.startswith("- NEW Y94 server1 pre-admission preparation"):
            line = ("- Y94 FIRST actual original native server1GPU0/350W PASS "
                    + evidence(y94["native_smoke_result"])
                    + "; losses[7.886719703674316,8.23161506652832], peak_reserved6515851264, "
                    "two finite TRAIN backward/optimizer steps, exact positions/hard exclusions/six "
                    "self-attention consumers observed. No CPU scientific replay or efficacy claim.")
        elif line.startswith("- Original Y94 model/YAML/CPU program"):
            line = ("- Y94 operational10/admission19 guards and Y93 lifecycle29/additive8 passed; "
                    "actual native evidence required, not a fixture. Receipt "
                    + evidence(ref("pipeline/admissions/AdaRES-Y94.json")) + "; full freeze "
                    + evidence(y94["post_training_freeze"])
                    + "; original complete Y92 training/evaluation inherited, candidate ID/version only.")
        elif line.startswith("- Production HEAD/native-trained identity"):
            line = ("- Production HEAD/native-trained identityY94=" + Y94
                    + "; observer y94_deployment.py observe --id AdaRES-Y93|AdaRES-Y94 "
                    "--token CURRENT --label NEW. Never observe archived Y92 or replay old observer at new HEAD.")
        elif line.startswith("- Y92 original PID"):
            line = ("- Y92 original PID1305468/server1GPU0/350W completed naturally; original directory"
                    "20261108_090202, Train/Val1..30 and stable finite last30/best20/model/optimizer544/"
                    "scheduler30-20 verified ONCE, canonically archived. Terminal "
                    + evidence(y92["terminal_evidence"]) + "; archive "
                    + evidence(y92["terminal_archive_binding"])
                    + ". Trainingbest .5092514753341675 is NOT formal FIRST. Never restart/re-finalize.")
        elif line.startswith("- Y93 original PID"):
            line = ("- Y93 original PID47380/server2GPU0/280W/code" + Y93
                    + "; exact original log/directory20261108_082521 retained. NEW actual-Y94-native "
                    "terminal compatibility " + evidence(y93["terminal_handoff_plan"])
                    + ";29 lifecycle+8 additive guards, onlyY93/Y94 markers. Future terminal helper "
                    "finalize_y93_fixed30_y94.py; no Y93 checkpoint loaded/stopped/archived this tick.")
        elif line.startswith("- Current AdaRES-Y92 "):
            continue
        elif line.startswith("- Current AdaRES-Y93 "):
            entry = observations["AdaRES-Y93"]
            phase = entry["phase"]
            line = ("- Current AdaRES-Y93 " + phase["phase"] + str(phase["epoch"])
                    + "/30 step" + str(phase["step"]) + "/" + str(phase["count"])
                    + "; original source/PID/native progress/280W verified: " + evidence(entry["evidence"]))
            lines.append(line)
            entry = observations["AdaRES-Y94"]
            phase = entry["phase"]
            line = ("- Current AdaRES-Y94 original PID218513/server1GPU0/350W, " + phase["phase"]
                    + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"])
                    + "; verified native counter increase/source/YAML/private runtime: "
                    + evidence(entry["evidence"])
                    + ". Original launch log train_adares_y94_query_geometry_bias_res_20261109_111829.log. "
                    "Do not infer checkpoint directory from launch timestamp; prepare NEW exact "
                    "terminal source/location/lifecycle binding before future archival.")
        elif line.startswith("- Fresh canonical occupation:"):
            line = ("- Fresh canonical occupation: "
                    + "; ".join(evidence(entry) for entry in occupations.values())
                    + "; both authorizedGPU0 busy/reserved, age<300s required. Preserve collector's actual tick_attempt.")
        elif line.startswith("- Next tick first observeY92/Y93"):
            line = ("- Next tick first observeY93/Y94 and dispatch ready work. Y95 thenY96 thenY98 "
                    "CPU-prepared/GPU-unexecuted/unadmitted; their HEADY93 bindings below are historical, "
                    "NOT executable on currentY94. Prepare NEW same-byte tested source/current-owner "
                    "bindings before exact matching natural30/full checkpoint/archive/fresh idleGPU0/native "
                    "execution. Preserve all CPU/version/native-plan evidence without replay; Y97 holds unchanged.")
        lines.append(line)
    where = lines.index("## Open Questions And Risks")
    lines.insert(where, "- Y92 original complete server1GPU0/350W full frozen review now waits onY94, "
                 "without restart/migration/substitution. Earlier reviews and GPU1 resource holds retain original devices.")
    lines += [
        "## Last Tick",
        "- " + token + ": Y92 naturally completed, exact original full finite checkpoint "
        "verification/canonical archive ONCE; no remote training stop/restart or old archive replay.",
        "- Actual original Y94 native YAML/TRAIN geometry/backward/two optimizer steps passed ONCE. "
        "Completed exact retained-source selection, full original freeze, receipt/admission and "
        "canonical safe dispatch; native increasing counters verified. Y93 original PID/runtime/"
        "protocol preserved with new29+8 actual-native additive terminal binding.",
        "- New scientificIDs0/scientificCPUattempts0/nativeGPUtasks1/admissions1/launches1/archives1. "
        "Y95/Y96/Y98 source/CPU/native plans remain intact but need NEW tested bindings onHEADY94; "
        "Y97 original/revision1 failures frozen, no extra revision or tolerance relaxation.",
        "- Audit wording clarification: advance_y94 inherited a visual quadratic readout sentence "
        "in its immutable admission decision template. Actual candidate/source/YAML/native evidence/"
        "receipt are bounded query-position self-attention bias, NOT Y92's quadratic mask readout. "
        "Separate corrective decision preserves the original signed text and hashes.",
        "- Goal bytes/old receipts/full9508x141 FIRST/twice/bootstrap10000 seed1999/TRAIN1487-3325/"
        "all harm metrics and original-device cost preserved. GPU1 all use and Y57 independently "
        "paused; no power/scheduler/server6/cross-project/foreign-job change or efficacy promotion.",
        "- Closeout " + evidence(ref(output))
        + "; exact memory backup, fresh sequential native observations and canonical physical occupation, "
        "collector actual resource_check retained. Official report/finish must verify "
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
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        assert before["pipeline/goal.json"]["sha256"] == GOAL
        assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == PAUSE
        index = git("ls-files", "--stage", "-z")
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
        original_refs = [
            dict(path="pipeline/admissions/AdaRES-Y92.json",
                 sha256="56c0a076cc22dd82200f934781dd50b7a5972ab2adb3c383acb42cb6cb0822dc"),
            dict(path="pipeline/admissions/AdaRES-Y93.json",
                 sha256="f6b87f2813f1e8e429772bc0f63963fe1a43e0582c6c65e21771092dcbf69832"),
        ]
        for entry in original_refs:
            closure.artifact(ROOT, entry)
        y92, y93, y94 = (candidates["AdaRES-" + name] for name in ("Y92", "Y93", "Y94"))
        candidate, _, _, frozen = prepared(work)
        assert candidate == y94 and y94["stage"] == "training"
        raw = tick.load(closure.artifact(ROOT, y94["native_smoke_result"]))
        validate_result(raw, y94, frozen)
        parent = tick.load(closure.artifact(ROOT, y92["post_training_freeze"]))
        review = tick.load(closure.artifact(ROOT, y94["post_training_freeze"]))
        expected = make_freeze(parent, y92["post_training_freeze"])
        for key in ("training", "evaluation", "resource_condition", "parent_protocol"):
            assert review[key] == expected[key]
        done = monitor.load_experiment(str(ROOT), "AdaRES-Y92")
        assert done["status"] == "done_complete" and done["code_commit"] == Y92
        terminal = tick.load(closure.artifact(ROOT, y92["terminal_evidence"]))
        validate_terminal(terminal, "server-1")
        assert terminal["checkpoints"][0]["epoch"] == 30
        assert terminal["checkpoints"][1]["epoch"] == 20
        archive = tick.load(closure.artifact(ROOT, y92["terminal_archive_binding"]))
        assert archive["id"] == "AdaRES-Y92" and archive["complete"] is True
        assert archive["checkpoint_files_retained"] and archive["remote_mutations"] is False
        assert y92["stage"] == "blocked" and y92["blocker_kind"] == "exclusive_gpu_slot"
        assert (y92["required_server"], y92["required_gpu"]) == ("server-1", 0)
        handoff = tick.load(closure.artifact(ROOT, y93["terminal_handoff_plan"]))
        assert handoff["CPU_guard_tests_passed"] == 29 and handoff["additive_guard_tests_passed"] == 8
        assert handoff["payload"]["allowed_markers"] == [Y93, Y94]
        assert handoff["tested_additive_native_evidence"] == y94["native_smoke_result"]
        assert handoff["post_training_freeze"] == y93["post_training_freeze"]
        assert handoff["supersedes"] == y93["pre_y94_terminal_handoff_plan"]
        assert handoff["checkpoint_loaded"] is False and handoff["archived"] is False
        assert handoff["remote_mutations"] is False and not y93.get("terminal_evidence")
        for entry in handoff["dependencies"]:
            closure.artifact(ROOT, entry)
        observations, occupations = {}, {}
        for name, cid, pid, commit in (
            ("server-1", "AdaRES-Y94", 218513, Y94), ("server-2", "AdaRES-Y93", 47380, Y93)
        ):
            task = next(row for row in queue["gpu_queues"][name] if row["id"] == cid)
            assert task["status"] == "running" and task["launch_verified"] is True
            assert (task["server"], str(task["gpu"]), task["code_commit"], task["pid"]) == (
                name, "0", commit, pid)
            closure.admission(ROOT, task)
            contract(commit, Y94, cid)
            entry = candidates[cid]["latest_training_observation"]
            observed = tick.load(closure.artifact(ROOT, entry))
            assert observed["token"] == token and observed["native_progress_confirmed"] is True
            assert observed["pid"] == pid and observed["source_marker"] in (commit, Y94)
            fresh(observed, "observed_local")
            observations[cid] = dict(evidence=entry, phase=observed["after"]["latest"], pid=pid)
            proof_ref = ref("pipeline/checks/gpu_occupation_" + name + "_release_" + token + ".json")
            proof = tick.load(closure.artifact(ROOT, proof_ref))
            fresh(proof, "observed_at")
            held = next(row for row in work["candidates"] if row.get("stage") == "blocked"
                and row.get("blocker_kind") == "exclusive_gpu_slot" and row.get("required_server") == name)
            snapshot_server = next(row for row in context["queue"]["servers"] if row["name"] == name)
            state = gpu_occupation.validated_state(ROOT, context, held, snapshot_server, queue)
            assert state["busy_gpu_ids"] == state["reserved_gpu_ids"] == [0]
            assert not state["idle_gpu_ids"]
            occupations[name] = proof_ref
        preparation = tick.load(closure.artifact(ROOT, y94["execution_preparation"]))
        assert preparation["operational_guards"] == 10
        assert preparation["actual_native_admission_guards"] == 19
        assert preparation["actual_native_evidence"] == y94["native_smoke_result"]
        for entry in preparation["evidence"]:
            closure.artifact(ROOT, entry)
        admission_tests = ref("pipeline/checks/adares-y94_admission_guards_" + token + ".log")
        text = closure.artifact(ROOT, admission_tests).read_bytes()
        assert b"Ran 19 tests" in text and b"OK" in text and b"skipped" not in text
        selection = ref("pipeline/checks/adares-y94_source_selection_" + token + ".json")
        selected = tick.load(closure.artifact(ROOT, selection))
        assert selected["parent"] == Y93 and selected["target"] == Y94
        assert selected["other_index_entries_preserved"] and selected["working_files_modified"] is False
        dispatch = ref("pipeline/checks/adares-y94_dispatch_" + token + ".json")
        launched = tick.load(closure.artifact(ROOT, dispatch))
        assert launched["task"]["launch_verified"] is True and launched["task"]["status"] == "running"
        assert launched["other_queue_entries_preserved"] is True
        untouched = {}
        for cid in ("AdaRES-Y95", "AdaRES-Y96", "AdaRES-Y98"):
            row = candidates[cid]
            assert row["stage"] == "blocked" and row["blocker_kind"] == "exclusive_gpu_slot"
            assert not row.get("native_smoke_result") and not row.get("admission_receipt")
            for key in ("CPU_result", "source_manifest", "native_preparation",
                        "native_smoke_plan_server1", "native_smoke_plan_server2_gpu0"):
                closure.artifact(ROOT, row[key])
            untouched[cid] = row["source_manifest"]
        numerical = candidates["AdaRES-Y97"]
        assert numerical["stage"] == "blocked" and numerical["blocker_kind"] == "numerical_control_failure"
        assert len(numerical["revision_history"]) == 1 and numerical["revision_history"][0]["result"] == "failed"
        closure.validate_revisions(ROOT, numerical)
        for key in ("CPU_failure", "CPU_program", "component_diagnosis", "revision1_result", "revision1_CPU_program"):
            closure.artifact(ROOT, numerical[key])
        output = ROOT / ("pipeline/checks/y94_execution_closeout_" + token + ".json")
        assert not output.exists()
        dependencies = original_refs + [
            ref(__file__), y92["terminal_evidence"], y92["terminal_archive_binding"],
            ref("pipeline/admissions/AdaRES-Y94.json"), y94["native_smoke_result"],
            y94["native_smoke"], y94["post_training_freeze"], y94["execution_preparation"],
            y93["terminal_handoff_plan"], selection, dispatch, admission_tests,
            numerical["CPU_failure"], numerical["revision1_result"],
        ] + [entry["evidence"] for entry in observations.values()] + list(occupations.values())
        tick.save(output, dict(
            complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            original_Y92_natural_fixed30_verified_and_canonically_archived=True,
            Y92_training_best_not_formal_FIRST=True, actual_Y94_native_pass=True,
            Y94_admitted=True, Y94_canonical_launch_and_increasing_native_counters=True,
            original_Y93_PID_preserved=True, Y93_actual_native_additive_terminal_binding=True,
            Y93_checkpoint_loaded=False, original_signed_receipts_and_full_freezes_preserved=True,
            prepared_unexecuted_candidates=untouched, Y97_original_and_revision1_failures_preserved=True,
            new_scientific_candidates=0, scientific_CPU_candidate_attempts=0, actual_native_GPU_tasks=1,
            new_admissions=1, new_training_launches=1, exact_original_archives=1,
            operational_guards=10, admission_guards=19, original_Y93_lifecycle_guards=29,
            actual_native_additive_guards=8, observations=observations, occupations=occupations,
            current_source_HEAD=Y94, full_original_FIRST_and_original_device_cost_preserved=True,
            all_GPU1_use_and_Y57_separately_paused=True, efficacy_measured=False,
            evidence=dependencies))
        y94["evidence"].append(ref(output))
        y94["next_action"] = (
            "Preserve original PID218513/server-1 GPU0/350W native fixed30 and observe increasing "
            "counters using y94_deployment.py. Before natural completion prepare a NEW exact native "
            "save-directory/source/YAML/receipt/freeze lifecycle binding without loading live checkpoints; "
            "never infer its save directory from launch time or use a stale terminal helper. "
            "Y93 original run continues; Y95/Y96/Y98 need NEW tested same-byte bindings on Y94. "
            "No CPU/native replay, restart, migration or scientific promotion from training metrics.")
        attempt = copy.deepcopy(work["tick_attempt"])
        assert attempt["kind"] == "resource_check"
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, lines = save_memory(token, view, candidates, observations, occupations, output)
        decision = monitor.log_decision(str(ROOT), "other",
            "Y92完整终态后，如何推进已通过真实原生检查的Y94并保留Y93？",
            "仅精确选择Y94已测七项新增文件，保留Y93原PID/协议，完成完整冻结收据、入队、canonical派发和递增原生counter验证。",
            "Y94是共享原始query位置的有界self-attention偏置，绝非静态mask二次读出；"
            "原不可变准入decision模板继承了后者描述，现另作文字澄清而不覆盖原收据或哈希。"
            "实际源/YAML/native检查/receipt均为Y94位置偏置。GPU1/Y57独立暂停，Y95/Y96/Y98原工件保留待新HEAD绑定。",
            output.relative_to(ROOT).as_posix())
        phases = []
        for cid, entry in observations.items():
            phase = entry["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30，步"
                + str(phase["step"]) + "/" + str(phase["count"]))
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅处理AdaRES_project；实际会话归属、Disabled后台调度、正式skill/experiment_first、"
            "AGENTS/goal/SERVERS和GPU1暂停核验，沿用本轮原租约/快照/deadline，先查进展派发现有任务。",
            "完成：Y92自然结束后只读精确核验原Train/Val1..30与稳定有限last30/best20/model/"
            "optimizer544/scheduler30-20，原目录20261108_090202，canonical归档一次，原产物及收据保留。"
            "训练best0.5092514753341675不是正式FIRST，也不宣布科学收益。",
            "新准备并启动：Y94原生YAML/真实TRAIN两步loss/backward/optimizer和原始位置/六层"
            "self-attention/硬mask路径实际通过，loss为7.886719703674316和8.23161506652832。"
            "完成10项操作、19项准入检查及Y93原运行29项生命周期+8项真实native加法绑定检查。"
            "仅选择已测七文件/保留其他index与working changes，原设备完整冻结协议/收据/入队/"
            "canonical安全同步派发均完成，原生训练步数增长已确认，不以PID代替启动。",
            "运行：" + "；".join(phases) + "。Y94为server1GPU0/原350W、Y93为server2GPU0/280W，"
            "Y93原PID47380/source/YAML/收据/私有环境/数据和检查点目录不变，未加载或归档Y93检查点。"
            "两卡canonical只读占用均busy+reserved，保持GPU1外部任务不动。",
            "保留Y95/Y96/Y98已测CPU/源码/native计划，不重放历史CPU/version/plan或已归档终态；"
            "当前HEAD已为Y94，后续执行前必须新建并测试同字节版本/当前owner绑定。Y97原始及"
            "修订1数值失败保持，无重试、阈值放宽或新修订。Y94精确终态位置/生命周期准备仍是后续动作。",
            "本轮新科学ID0、科学CPU尝试0、真实GPU原生1、准入1、训练启动1、精确归档1。"
            "完整9508表达/141场景/candidate+Y36-W2-B1各两次/FIRST主报告/场景bootstrap10000-"
            "seed1999/TRAIN1487-3325/全部有害子组与指标/原设备cost协议保持。"
            "已精确备份七节记忆，保留collector实际resource_check，再正式report/finish，不预称收尾PASS。"
        ], decisions=[
            "优先完成已有Y94的真实native到本轮入队和启动，不为凑额度新增或复制机制。"
            "后续观察Y93/Y94；Y93自然30轮仅使用新finalize_y93_fixed30_y94.py精确终态链。"
            "Y94先准备原始声明保存目录的新终态链，不从启动时间猜目录，不加载live checkpoint。",
            "Y92完整原server1GPU0/350W冻结后评等待Y94；原GPU1冻结后评继续挂起，"
            "不迁移或替代资源。科学goal字节、原收据、原数据/私有runtime/mappings、固定30轮seed1999、"
            "禁用项、专用恢复要求、功耗和调度不变。server2GPU1全部使用和Y57实验分别暂停，"
            "禁止server6/跨项目/外部任务干预，无新用户操作需求。",
            "审计文字澄清：不可变advance_y94准入decision模板继承了Y92静态mask二次读出一句，"
            "并非实际执行机制；源代码、独立YAML、native路径和签名receipt均是Y94有界query位置"
            "self-attention偏置。原文和哈希保留，本条与独立decision明确纠正，不伪改历史证据。"
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
    print(json.dumps(dict(
        closeout=ref(output), observations=tick.load(output)["observations"],
        status=receipt["status"], postflight=receipt["postflight"], context_committed=True,
        lease_release=receipt["lease_release"], memory_lines=lines, archive_pending_files=0,
        new_admissions=1, launches=1, archives=1, notes=ref(notes)), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    finish(parser.parse_args().token)
