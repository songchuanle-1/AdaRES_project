"""Close new pending-operation checks without claiming native or launch progress."""
import argparse
from datetime import datetime
import json
import os
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y93_deployment import Y92, Y93, contract, git
import closure
import gpu_occupation


def evidence(entry):
    return entry["path"] + " SHA" + entry["sha256"]


def memory(token, view, candidate, observations, occupations, output):
    path = ROOT / "pipeline/codex_memory.md"
    backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
    assert not backup.exists()
    original = path.read_bytes()
    backup.write_bytes(original)
    assert backup.read_bytes() == original
    old = original.decode("utf-8").splitlines()
    lines = old[:old.index("## Last Tick")]
    for index, line in enumerate(lines):
        if line.startswith("- Current token "):
            lines[index] = "- Current token " + token + "; snapshot " + view["generated_at"] + "; deadline " + view["deadline_at"] + "; never extend."
        elif line.startswith("- Exact previous seven-section memory:"):
            lines[index] = "- Exact previous seven-section memory: " + evidence(ref(backup)) + "; complete prior snapshot-chain history/bans/negative evidence/dedicated recovery requirements retained byte-for-byte."
        elif line.startswith("- Current AdaRES-Y92 ") or line.startswith("- Current AdaRES-Y93 "):
            cid = line.split()[2]
            entry = observations[cid]
            phase = entry["phase"]
            lines[index] = "- Current " + cid + " " + phase["phase"] + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"]) + "; original native progress/source/YAML/runtime/power verified: " + evidence(entry["evidence"])
        elif line.startswith("- Fresh canonical occupation:"):
            lines[index] = "- Fresh canonical occupation: " + "; ".join(evidence(entry) for entry in occupations.values()) + "; both permitted GPU0 busy/reserved, age<300s required. Keep collector's actual tick_attempt and all candidate evidence."
        elif line.startswith("- Next tick first observeY92/Y93"):
            lines[index] = "- Next tick first observeY92/Y93 and dispatch ready work; Y94 thenY95 thenY96 thenY98 remain CPU-prepared/GPU-unexecuted/unadmitted. Y94 server1 pre-admission checks are prepared separately below; native activation remains server2 unchanged. Exact matching owner natural30/full finite checkpoint/archive and fresh idle GPU0 still required. Y97 failures stay immutable; no timer CPU/version/plan/terminal replay; changed HEAD/owner needs NEW tested bindings."
    where = next(index for index, line in enumerate(lines) if line.startswith("- Y94 CURRENT")) + 1
    lines[where:where] = [
        "- NEW Y94 server1 pre-admission preparation " + evidence(candidate["server1_pre_admission_preparation"]) + ":17 new actual-artifact source/private-runtime/current-owner/missing-native/full-original-protocol guards passed first execution, no native-result fixtures. Helper scripts/y94_server1_pre_admission.py is read-only and rejects missing actual Y94 native evidence. Full original Y92/server1/350W training/review inheritance tested IN MEMORY ONLY, not a registered Y94 freeze, native PASS or admission.",
        "- Original Y94 model/YAML/CPU program/result/both native plans/current activation/source commit stay unchanged; no CPU scientific/version/plan test replay. After unchanged run_y94_bound_native_y93.py --server server-1 actual PASS, use new read-only verifier BEFORE selection. Still prepare a NEW tested Y93 additive terminal binding using actual Y94 native evidence, then exact source selection/full freeze/receipt/admission/canonical dispatch/increasing counters; full launch toolchain is NOT yet prepared. Original server2 alternative needs its own matching future execution path.",
    ]
    lines += [
        "## Last Tick",
        "- " + token + ": original Y92/Y93 native progress/source/YAML/receipts/private runtimes/350W-280W verified. No terminal, restart, checkpoint loading, stop, migration or archive. Both permittedGPU0 busy/reserved; Y94..Y98 GPU checks cannot displace them.",
        "- Advanced earlier Y94 with17 new pending server1 operation guards using actual retained sources/CPU evidence/native plans/current owners/original Y92 full freeze; missing native result and still-running Y92 rejected. Source/activation/plans and other prepared variants remain unchanged. No synthetic GPU native-result fixture or CPU scientific replay; protocol draft was memory-only.",
        "- New scientific IDs0/scientific CPU attempts0/new operational test suite1(17 tests passed)/GPU native0/admissions0/launches0/archives0. Y97 original/revision1 numerical holds remain untouched; no extra revision, renamed duplicate, FIRST promotion or new user permission request.",
        "- All scientific goal bytes/old receipts/full original FIRST and original-device cost/data/runtime/mappings/seed1999-fixed30/bans/power/schedule/restore rules remain unchanged. GPU1 all use and original GPU1 frozen review stay paused; Y57 independently paused; no resource substitution/foreign-job interference.",
        "- Closeout " + evidence(ref(output)) + "; exact memory backup, sequential fresh native observations/canonical occupation, preserve collector's actual resource_check, official Chinese report/finish. No preclaim; verify FINALIZED/PASS/context_committed/released and absent lock/pending/archive journals read-only after finish.",
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
        assert git("rev-parse", "HEAD").decode().strip() == Y93
        before = {path: ref(path) for path in ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        assert before["pipeline/goal.json"]["sha256"] == GOAL
        assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == PAUSE
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {row["id"]: row for row in work["candidates"]}
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        context = tick.load(ROOT / "pipeline/headless/.tick_context.json")
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        assert running == {"server-1": {"0": "AdaRES-Y92"}, "server-2": {"0": "AdaRES-Y93"}}
        servers = monitor.load_servers(str(ROOT))
        assert servers["server-1"]["allowed_gpus"] == servers["server-2"]["allowed_gpus"] == [0]
        assert servers["server-6"]["allowed_gpus"] == []
        paused = next(row for row in queue["gpu_queues"]["server-2"] if row["id"] == "AdaRES-Y57")
        assert paused["status"] == "paused" and paused["dispatch_hold"] and paused["resume_requires_user_authorization"]
        observations, occupations = {}, {}
        for name, cid, pid, commit in (
            ("server-1", "AdaRES-Y92", 1305468, Y92), ("server-2", "AdaRES-Y93", 47380, Y93),
        ):
            task = next(row for row in queue["gpu_queues"][name] if row["id"] == cid)
            assert task["status"] == "running" and task["launch_verified"] is True and task["pid"] == pid
            assert task["server"] == name and str(task["gpu"]) == "0" and task["code_commit"] == commit
            closure.admission(ROOT, task)
            contract(commit, Y93, cid)
            entry = candidates[cid]["latest_training_observation"]
            observed = tick.load(closure.artifact(ROOT, entry))
            assert observed["token"] == token and observed["native_progress_confirmed"] is True
            assert observed["pid"] == pid and observed["source_marker"] == commit
            fresh(observed, "observed_local")
            observations[cid] = dict(evidence=entry, phase=observed["after"]["latest"], pid=pid)
            proof_ref = ref("pipeline/checks/gpu_occupation_" + name + "_release_" + token + ".json")
            proof = tick.load(closure.artifact(ROOT, proof_ref))
            fresh(proof, "observed_at")
            held = next(row for row in work["candidates"] if row.get("stage") == "blocked"
                and row.get("blocker_kind") == "exclusive_gpu_slot" and row.get("required_server") == name)
            snapshot_server = next(row for row in context["queue"]["servers"] if row["name"] == name)
            state = gpu_occupation.validated_state(ROOT, context, held, snapshot_server, queue)
            assert state["busy_gpu_ids"] == state["reserved_gpu_ids"] == [0] and not state["idle_gpu_ids"]
            occupations[name] = proof_ref
            terminal = tick.load(closure.artifact(ROOT, candidates[cid]["terminal_handoff_plan"]))
            assert terminal["checkpoint_loaded"] is False and terminal["archived"] is False
            assert terminal["post_training_freeze"] == candidates[cid]["post_training_freeze"]
        pending = {}
        for cid in ("AdaRES-Y94", "AdaRES-Y95", "AdaRES-Y96", "AdaRES-Y98"):
            row = candidates[cid]
            assert row["stage"] == "blocked" and row["blocker_kind"] == "exclusive_gpu_slot"
            assert not row.get("native_smoke_result") and not row.get("admission_receipt")
            for key in ("CPU_result", "source_manifest", "native_preparation",
                        "native_smoke_plan_server1", "native_smoke_plan_server2_gpu0"):
                closure.artifact(ROOT, row[key])
            pending[cid] = row["source_manifest"]
        numerical = candidates["AdaRES-Y97"]
        assert numerical["stage"] == "blocked" and numerical["blocker_kind"] == "numerical_control_failure"
        assert len(numerical["revision_history"]) == 1 and numerical["revision_history"][0]["result"] == "failed"
        closure.validate_revisions(ROOT, numerical)
        for key in ("CPU_failure", "CPU_program", "component_diagnosis", "revision1_result", "revision1_CPU_program"):
            closure.artifact(ROOT, numerical[key])
        row = candidates["AdaRES-Y94"]
        plan_ref = row["server1_pre_admission_preparation"]
        preparation = tick.load(closure.artifact(ROOT, plan_ref))
        assert preparation["token"] == token and preparation["working_HEAD"] == Y93
        assert preparation["pending_operation_tests_passed"] == 17
        assert preparation["native_GPU_executed"] is False
        assert preparation["CPU_scientific_tests_replayed"] is False
        assert preparation["native_result_fixture_created"] is False
        assert preparation["formal_freeze_registered"] is False and preparation["full_launch_toolchain_prepared"] is False
        assert preparation["original_candidate_activation_preserved"] is True
        assert preparation["both_original_native_plans_preserved"] is True
        for entry in preparation["evidence"]:
            closure.artifact(ROOT, entry)
        log = ROOT / ("pipeline/checks/y94_server1_pre_admission_" + token + ".log")
        raw = log.read_bytes()
        assert b"Ran 17 tests" in raw and b"OK" in raw and b"skipped" not in raw
        assert row["required_server"] == "server-2" and row["required_gpu"] == 0
        assert row["native_smoke_plan"] == row["native_smoke_plan_server2_gpu0"]
        assert not (ROOT / "pipeline/admissions/AdaRES-Y94.json").exists()
        output = ROOT / ("pipeline/checks/y94_server1_pre_admission_closeout_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            observations=observations, occupations=occupations, prepared_candidates=pending,
            operational_preparation=plan_ref, new_operational_tests=17,
            full_original_protocol_in_memory_only=True, native_result_fixture_created=False,
            all_original_CPU_versions_native_plans_and_activation_preserved=True,
            original_Y97_numerical_failures_preserved=True, original_Y92_Y93_preserved=True,
            new_scientific_candidates=0, scientific_CPU_candidate_attempts=0, native_GPU_tasks=0,
            new_admissions=0, new_training_launches=0, original_archives=0,
            original_training_checkpoints_loaded=False, formal_freeze_registered=False,
            full_launch_toolchain_prepared=False, original_full_frozen_FIRST_and_cost_preserved=True,
            original_GPU1_and_Y57_pause_preserved=True, efficacy_measured=False,
            production_HEAD_index_queue_preserved=True,
            evidence=[ref(__file__), plan_ref, ref(log), numerical["CPU_failure"], numerical["revision1_result"]]
                + [entry["evidence"] for entry in observations.values()] + list(occupations.values())))
        row["evidence"].append(ref(output))
        assert work["tick_attempt"]["kind"] == "resource_check"
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, lines = memory(token, view, row, observations, occupations, output)
        decision = monitor.log_decision(str(ROOT), "other",
            "已有Y94/Y95/Y96/Y98通过CPU但授权GPU0仍忙，优先补齐哪个可执行环节？",
            "优先为较早Y94准备server1准入前17项新边界检查，保持当前server2激活和双原生计划不变；真实GPU/native/完整终态/后续源选择与准入仍待满足。",
            "新检查基于实际已测七文件、私有环境/350W、原Y92完整冻结协议和当前owner；无原生结果或Y92仍运行均拒绝。CPU科学测试未重放，无模拟native结果，无新科学ID/队列/收据/冻结注册或启动。",
            output.relative_to(ROOT).as_posix())
        phases = []
        for cid, entry in observations.items():
            phase = entry["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30，步"
                + str(phase["step"]) + "/" + str(phase["count"]))
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅处理AdaRES_project；owner实际会话、Disabled后台调度、正式skill/experiment_first、AGENTS/goal/SERVERS及GPU1暂停核验后begin。先查进展并尝试已有合格派发，ready waiting为0，无新终态。",
            "运行：" + "；".join(phases) + "；原PID/source/YAML/receipt/私有runtime、350W/280W及原生递增步数确认。canonical物理占用确认两个授权GPU0 busy+reserved，不干扰GPU1/外部进程。",
            "新准备：较早候选Y94的server1准入前边界检查，17项新测试首次实际执行通过；使用实际七文件/冻结native payload、当前资源owner、原Y92完整训练与9508表达141场景FIRST/双重复/全部指标/场景bootstrap10000-seed1999/原TRAIN切分/有害子组和同设备cost协议。",
            "检查在内存中验证完整原协议继承，父对象绑定原哈希工件，不注册Y94冻结计划；真实native结果缺失、当前Y92未完整归档、错误源码/HEAD/goal/设备/功耗/协议等均拒绝。无native结果fixture，不冒充实际GPU两步通过或完整启动工具链完成。",
            "原Y94模型/YAML/科学CPU结果/留版/双原生计划和server2激活不变；Y95/Y96/Y98与Y97原始及修订1数值失败保持。CPU科学/version/plan测试和已归档终态不重放。当前两项训练终态计划仅核验工件，未加载checkpoint、停止或归档。",
            "本轮新科学ID0、科学CPU尝试0、新操作测试组1计17项通过、GPU原生0、准入0、启动0、归档0。先精确备份七节记忆、刷新训练与canonical占用、保留collector实际resource_check，再正式report/finish，不预先宣称收尾结果。"
        ], decisions=[
            "先推进已有较早候选，不为凑额度再造变体；本轮仅完成Y94缺失的操作边界准备，不改候选科学控制或当前资源激活。Y94/Y95/Y96/Y98均待真实GPU原生通过，不能伪造waiting或native PASS。",
            "在HEADY93、Y92自然30轮/完整有限checkpoint精确核验归档及实时空闲server1GPU0后运行原run_y94_bound_native_y93.py；实际native通过后可用新只读verifier核对。源选择前仍须新的、实测Y94 native证据绑定的Y93终态兼容检查；然后完成精确源选择/完整freeze/receipt/准入/安全派发和递增counter。server2替代路径须匹配自身owner/未来执行链，不擅自迁移。",
            "科学goal字节、原签名收据、数据/私有环境/mappings/seed1999固定30轮/禁用/恢复/功耗/调度与完整FIRST同设备cost协议不变。GPU1全部使用及原GPU1冻结后评继续暂停，Y57独立暂停；不用server6，不跨项目，不驱逐或替代资源。无需新用户操作。"
        ], evidence=[ref(output), backup, ref(decision)]))
        assert all(ref(path) == value for path, value in before.items())
        return output, notes, lines
    finally:
        writer.close()


def finish(token):
    lease(token, 240)
    for cid in ("AdaRES-Y92", "AdaRES-Y93"):
        command(["scripts/y93_deployment.py", "observe", "--id", cid, "--token", token, "--label", "release"])
    for name in ("server-1", "server-2"):
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
    assert not pending.exists() or not any(path.is_file() for path in pending.rglob("*"))
    assert ref("pipeline/goal.json")["sha256"] == GOAL
    print(json.dumps(dict(closeout=ref(output), observations=tick.load(output)["observations"],
        receipt_status=receipt["status"], postflight=receipt["postflight"], context_committed=True,
        lease_release=receipt["lease_release"], memory_lines=lines, archive_pending_files=0, notes=ref(notes)),
        ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    finish(parser.parse_args().token)
