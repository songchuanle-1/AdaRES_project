"""Close an honestly failed CPU realization with original controls preserved."""
import argparse
import ast
import base64
from datetime import datetime
import json
import os
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y93_deployment import Y92, Y93, contract, git
from prepare_y97_language_quadratic_input import files
import closure
import gpu_occupation


def memory(token, view, candidates, observations, occupations, output):
    path = ROOT / "pipeline/codex_memory.md"
    backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
    assert not backup.exists()
    original = path.read_bytes()
    backup.write_bytes(original)
    assert backup.read_bytes() == original
    old = original.decode("utf-8").splitlines()
    lines = old[:old.index("## Last Tick")]
    new = candidates["AdaRES-Y97"]

    def evidence(entry):
        return entry["path"] + " SHA" + entry["sha256"]

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
            lines[index] = "- Next tick first observeY92/Y93 and dispatch ready work; three GPU-unexecuted/unadmitted prepared candidates Y94 thenY95 thenY96. Matching original-owner natural30/full checkpoint/archive plus fresh idle GPU0 required. Y97 has a separate numerical implementation hold, not usable reserve or human/GPU permission. No CPU/plan/archived-terminal replay; changed HEAD/owner requires NEW tested bindings."
        elif "No new failure/repair this tick." in line:
            lines[index] = line.replace("No new failure/repair this tick.", "Historical repairs preserved; current Y97 CPU numerical failure is separately recorded.")
    where = lines.index("## Active Hypotheses") + 1
    lines[where:where] = [
        "- Y97 independent W2 full-token quadratic language input: L=original_lang_norm(original_lang_proj(T)+T.square()@zeroW768x256), once; +196608/total164826245/no RNG. Original full tokens/channels/norm/epsilon/masks/frozen text encoder/fullinputs/2D/backbone/sampling algorithm/SCG/hard-gate algorithm/static masks/six refinements/shared heads/targets-loss-selector/seed1999/fixed30/native YAML unchanged. Not pooled language-queryY75, recurrent feedbackY79, final-confidenceY33/Y35, Y23 loss padding, rejectedY26 norm/Y53 padding premises or a combination.",
        "- Y97 FIRST actual CPU original model/pretrain/TRAIN initialization reached but FAILED component exact batch-permutation guard torch.equal(layer(projected.flip(0),raw.flip(0)),value.flip(0)). Full decoder anchors, shared observer callbacks and two synthetic updates were NOT executed. Original model/YAML/program/log remain unchanged; no CPU PASS/retained code commit/native plan/GPU/native/admission/efficacy.",
        "- Y97 unchanged-guard component diagnosis reproduced both failures: float32 max1.1920928955078125e-7,64/70 exact; float64 max2.220446049250313e-16,69/70 exact. Both are within existing independent scalar rtol/atol128eps, but this is descriptive evidence ONLY, never a replacement PASS or permission to relax equality. CPU-only component execution, no fullmodel/data replay or CUDA.",
        "- Y97 failure " + evidence(new["CPU_failure"]) + "; diagnosis " + evidence(new["component_diagnosis"]) + "; original CPU program " + evidence(new["CPU_program"]) + ". Stable Y97 stageblocked/numerical_control_failure/failure_scopeimplementation, no revision used and no mechanism rejection. Preserve frozen guards/all negative evidence; no timer-based replay/renaming. Only an evidence-backed SAME-ID bounded implementation revision, preregistered BEFORE test, may attempt unchanged full controls.",
    ]
    lines += [
        "## Last Tick",
        "- " + token + ": original Y92/Y93 progressing/source/YAML/receipts/private runtimes/350W-280W verified; no new terminal. EarlierY94/Y95/Y96 exact CPU versions and native plans preserved without replay; Y92/Y93 terminal plans only hash-checked, no checkpoints loaded/stopped/archived.",
        "- New Y97 full-token quadratic input implementation; FIRST actual CPU failed exact component batch-permutation guard before full decoder tests. Separate component-only diagnosis repeats SAME equality and measures1epsilon float32/64 differences. Original guard/model/YAML untouched, no repair/revision/PASS/admission; implementation numerical hold, NOT falsified hypothesis.",
        "- New scientific IDs1/actual CPU candidate attempts1(failed)/small CPU component diagnostics1/GPU native0/admissions0/launches0/archives0. Later contract/tests/retainer source drafts remain unexecuted and do not imply checks passed. No new request for user authorization.",
        "- All scientific goal bytes/old receipts/full original FIRST and same-device cost/data/runtime/mappings/seed1999-fixed30/bans/power/schedule/restore rules preserved. GPU1 all use and Y57 separate pause remain; no migration/substitution/foreign-job interference.",
        "- Closeout " + evidence(ref(output)) + "; sequential current native observations/canonical physical occupation, keep collector's actual resource_check, then official report/finish. No preclaim; after finish read only FINALIZED/PASS/context_committed/released and absent lock/pending/archive journals. No post-release writes.",
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
        earlier = {}
        for cid in ("AdaRES-Y94", "AdaRES-Y95", "AdaRES-Y96"):
            candidate = candidates[cid]
            assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
            assert not candidate.get("native_smoke_result") and not candidate.get("admission_receipt")
            for key in ("CPU_result", "source_manifest", "native_preparation"):
                closure.artifact(ROOT, candidate[key])
            earlier[cid] = candidate["source_manifest"]
        new = candidates["AdaRES-Y97"]
        assert new["stage"] == "blocked" and new["blocker_kind"] == "numerical_control_failure"
        assert new["failure_scope"] == "implementation" and not new.get("revision_history")
        assert not any(new.get(key) for key in ("code_commit", "CPU_result", "source_manifest", "native_preparation",
            "native_smoke_result", "admission_receipt"))
        failure = tick.load(closure.artifact(ROOT, new["CPU_failure"]))
        diagnosis = tick.load(closure.artifact(ROOT, new["component_diagnosis"]))
        assert failure["original_guard_unchanged"] and failure["original_model_and_YAML_unchanged"]
        assert failure["original_CPU_attempt"]["full_CPU_control_suite_passed"] is False
        assert failure["original_CPU_attempt"]["decoder_backward_executed"] is False
        assert diagnosis["original_float32_failure_reproduced"] and not diagnosis["CUDA_initialized"]
        payload = ast.literal_eval(ast.parse(closure.artifact(ROOT, new["CPU_program"]).read_text(encoding="utf-8")).body[0].value)
        for path in files():
            value = payload["variant"]["config_data"] if path.suffix == ".yaml" else payload["variant"]["python_files"][path.name]
            assert path.read_bytes() == base64.b64decode(value["base64"]) and ref(path)["sha256"] == value["sha256"]
        for entry in failure["evidence"]:
            closure.artifact(ROOT, entry)
        output = ROOT / ("pipeline/checks/y97_numerical_closeout_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            observations=observations, occupations=occupations, earlier_preparations=earlier,
            Y97_original_failure=new["CPU_failure"], Y97_component_diagnosis=new["component_diagnosis"],
            original_guard_model_YAML_preserved=True, no_repair_or_revision_executed=True,
            original_Y92_Y93_and_earlier_Y94_Y95_Y96_preserved=True,
            original_full_frozen_FIRST_and_cost_preserved=True, original_GPU1_and_Y57_pause_preserved=True,
            new_scientific_candidates=1, CPU_candidate_attempts=1, CPU_candidate_attempt_result="failed",
            CPU_component_only_diagnostics=1, native_GPU_tasks=0, new_admissions=0,
            new_training_launches=0, original_archives=0, original_training_checkpoints_loaded=False,
            mechanism_falsified=False, efficacy_measured=False, production_HEAD_index_queue_preserved=True,
            evidence=[ref(__file__), new["CPU_failure"], new["CPU_program"], new["component_diagnosis"]]
                + [row["evidence"] for row in observations.values()] + list(occupations.values())))
        new["evidence"].append(ref(output))
        assert work["tick_attempt"]["kind"] == "resource_check"
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, lines = memory(token, view, candidates, observations, occupations, output)
        decision = monitor.log_decision(str(ROOT), "other",
            "Y97首次CPU检查在精确批次置换逐位守卫失败，如何保留数值事实？",
            "保存原程序/日志和相同守卫诊断；Y97按numerical_control_failure/implementation挂起，不准入、不弱化守卫、不否定整个假设。",
            "重现float32最大1.1920928955078125e-7、float64最大2.220446049250313e-16，虽在已有独立标量128eps容差内，但逐位相等原条件仍失败；未完成后续decoder/反向/native检查。",
            output.relative_to(ROOT).as_posix())
        phases = []
        for cid, entry in observations.items():
            phase = entry["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30，步"
                + str(phase["step"]) + "/" + str(phase["count"]))
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅处理AdaRES_project；已核验实际owner会话、Disabled后台调度、正式skill/experiment_first、AGENTS/goal/SERVERS及最新GPU1暂停记录。begin先查进展并尝试已有合格派发；初始ready waiting为0，无新终态。",
            "运行：" + "；".join(phases) + "；原PID/source/YAML/receipt/私有runtime、350W/280W及原生步数递增核实，canonical物理占用确认两GPU0 busy+reserved。GPU1及外部进程未干扰。",
            "新实现Y97独立全token语言输入二次投影，保留原投影及同一lang_norm；原YAML、完整CPU模型和发布骨干/TRAIN初始化后，小组件批次置换torch.equal守卫首次实际执行失败，未生成CPU通过结果。",
            "保存原失败程序和raw日志；单独CPU小组件诊断重复相同未改守卫，float32最大差1.1920928955078125e-7、64/70逐位相等，float64最大差2.220446049250313e-16、69/70逐位相等。虽处于原独立标量128eps容差内，但只作诊断证据，不能替代逐位守卫PASS。",
            "Y97按实现数值问题挂起，不放宽检查、不修改原模型/YAML、不进行修复或科学修订、不重命名重开。后续完整decoder锚点、共享native回调、两步反向均未执行；未留版、未生成native计划、未GPU检查、未准入/同步/派发。后创建的contract/tests/retainer源码均未执行，不冒充18项守卫通过。",
            "较早Y94/Y95/Y96完整CPU准备/独立版本/native计划保持，不重复科学检查；原Y92/Y93终态计划只读核对哈希，未加载checkpoint或执行归档，已归档实验不重放。",
            "本轮新增科学候选1、CPU候选检查1失败、CPU小组件诊断1、GPU原生0、准入0、启动0、归档0。七节记忆先精确备份，连续刷新当前训练及canonical物理占用后正式report/finish；保留collector实际resource_check，不预先宣称收尾结果。"
        ], decisions=[
            "Y97是implementation数值守卫失败，不是训练无效、缺用户授权或假设被反证。保留全部原证据与原检查；未来仅同一ID下事前登记、证据支持且预算内的局部实现修订可重测原完整控制，不凭计时重跑或改容差。",
            "原Y92/Y93继续。Y94、Y95、Y96按原优先顺序等待匹配原owner自然30轮/完整有限checkpoint精确核验归档及实时独占空闲GPU0，然后真实native通过才按原freeze/admission/安全同步和递增counter推进。Y97挂起不阻断其他工作。",
            "完整原FIRST/同设备cost、数据/私有环境/mappings/seed1999-fixed30/有害子组/禁用/功耗/调度/恢复要求、科学goal字节和原receipt保持。GPU1全部使用与原GPU1后评/cost继续暂停，Y57另行暂停，不迁移替代，禁止server6；无需新用户操作。"
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
