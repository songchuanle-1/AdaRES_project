"""Close the failed same-ID revision without rewriting any original controls."""
import argparse
import ast
import base64
from datetime import datetime
import json
import os
import re
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y93_deployment import Y92, Y93, contract, git
from revise_y97_tokenwise_projection import files, OLD, NEW
import closure
import gpu_occupation


def evidence(entry):
    return entry["path"] + " SHA" + entry["sha256"]


def memory(token, view, row, observations, occupations, output):
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
        elif line.startswith("- Y97 failure "):
            lines[index] = line.replace("no revision used and no mechanism rejection.",
                "original failure preserved; the later SAME-ID revision1 is separately recorded below, no mechanism rejection.")
        elif line.startswith("- Current AdaRES-Y92 ") or line.startswith("- Current AdaRES-Y93 "):
            cid = line.split()[2]
            entry = observations[cid]
            phase = entry["phase"]
            lines[index] = "- Current " + cid + " " + phase["phase"] + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"]) + "; original native progress/source/YAML/runtime/power verified: " + evidence(entry["evidence"])
        elif line.startswith("- Fresh canonical occupation:"):
            lines[index] = "- Fresh canonical occupation: " + "; ".join(evidence(entry) for entry in occupations.values()) + "; both permitted GPU0 busy/reserved, age<300s required. Keep collector's actual tick_attempt and all candidate evidence."
        elif line.startswith("- Next tick first observeY92/Y93"):
            lines[index] = "- Next tick first observeY92/Y93 and dispatch ready work; Y94 thenY95 thenY96 remain CPU-prepared, GPU-unexecuted/unadmitted. Matching owner natural30/full checkpoint/archive plus fresh idle GPU0 required. Y97 original+revision1 numerical failures are immutable, not usable reserve or missing permission. No timed CPU/plan/terminal replay; changed HEAD/owner needs NEW tested bindings."
    where = lines.index("## Active Hypotheses") + 1
    lines[where:where] = [
        "- Y97 SAME-ID implementation revision1 preregistered BEFORE execution: replace only collapsed-row quadratic GEMM with fixed-shape per-token torch.mv, stack and add the SAME original projected tokens. +196608/total164826245/no RNG/full tokens/channels/norm/protocol unchanged. Separate packageadares_y97_language_quadratic_input_rev1 preserves every original model/YAML path byte-for-byte; not a new mechanism or renamed retry.",
        "- Revision1 registration " + evidence(row["revision1_registration"]) + "; preparation " + evidence(row["revision1_preparation"]) + "; new program " + evidence(row["revision1_CPU_program"]) + ". Entire CPU program AFTER payload and original component/native callback guards are unchanged. Full original raw log retained; no tolerance/equality/threshold change.",
        "- Revision1 FIRST actual full CPU attempt FAILED at capture_adapter original torch.equal(value, projected + raw.square() @ module.weight). Fail-fast position shows component float32/64 controls, seven-stage zero anchors/input-gradient checks and causal bypass were reached earlier, not that the complete suite/two updates passed. No completed CPU result, native preparation, retained commit, GPU task, admission or efficacy. Result " + evidence(row["revision1_result"]) + ".",
        "- Y97 remains stageblocked/numerical_control_failure/failure_scopeimplementation, revision1 resultfailed; original failure/diagnosis/model/program/log and this revision remain immutable. Do not replay either, switch to tolerance, loosen the original batched-matmul equality, rename, or claim the hypothesis falsified. Further work needs new evidence and a bounded preregistered SAME-ID realization that preserves ALL original controls; no user permission request is inferred.",
    ]
    lines += [
        "## Last Tick",
        "- " + token + ": existing Y92/Y93 native progress/source/YAML/receipts/private runtimes/350W-280W verified. No terminal, restart, checkpoint loading, migration or archive. EarlierY94/Y95/Y96 CPU/native plans preserved without replay; original GPU1 frozen review and Y57 remain paused.",
        "- Y97 SAME-ID revision1 was registered, implemented in a separate package and actually CPU-tested once with the entire original fail-fast controls unchanged. It passed the earlier component batch-permutation location but failed the later full-consumer batched-matmul bitwise guard. Complete CPU PASS/two-update PASS/native PASS are NOT established; no further repair or second revision executed.",
        "- New scientific IDs0/same-ID local revisions1/actual full CPU attempts1(failed)/GPU native0/admissions0/launches0/archives0. No training-best/FIRST promotion or new operation authorization requested. Original signed receipts and scientific goal bytes unchanged.",
        "- Closeout " + evidence(ref(output)) + "; sequential fresh native observations/canonical physical occupation, memory backed up exactly, Chinese decision/notes, official report/finish. Keep collector's actual resource_check. No preclaim; verify FINALIZED/PASS/context_committed/released and absent lock/pending/archive journals read-only after finish.",
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
            row = candidates[cid]
            assert row["stage"] == "blocked" and row["blocker_kind"] == "exclusive_gpu_slot"
            assert not row.get("native_smoke_result") and not row.get("admission_receipt")
            for key in ("CPU_result", "source_manifest", "native_preparation"):
                closure.artifact(ROOT, row[key])
            earlier[cid] = row["source_manifest"]
        row = candidates["AdaRES-Y97"]
        assert row["stage"] == "blocked" and row["blocker_kind"] == "numerical_control_failure"
        assert row["failure_scope"] == "implementation" and len(row["revision_history"]) == 1
        assert row["revision_history"][0]["result"] == "failed"
        assert not any(row.get(key) for key in ("code_commit", "CPU_result", "source_manifest",
            "native_preparation", "native_smoke_result", "admission_receipt"))
        registration = tick.load(closure.artifact(ROOT, row["revision1_registration"]))
        closure.validate_revisions(ROOT, row, registration["original_candidate"])
        for entry in registration["original_candidate"]["evidence"]:
            closure.artifact(ROOT, entry)
        for entry in registration["original_files"]:
            closure.artifact(ROOT, entry)
        original_path = closure.artifact(ROOT, registration["original_CPU_program"])
        original_text = original_path.read_text(encoding="utf-8")
        revised_path = closure.artifact(ROOT, row["revision1_CPU_program"])
        revised_text = revised_path.read_text(encoding="utf-8")
        assert revised_text.partition("\n")[2] == original_text.partition("\n")[2]
        payload = ast.literal_eval(ast.parse(revised_text).body[0].value)
        original_payload = ast.literal_eval(ast.parse(original_text).body[0].value)
        for path in files():
            value = payload["variant"]["config_data"] if path.suffix == ".yaml" else payload["variant"]["python_files"][path.name]
            assert path.read_bytes() == base64.b64decode(value["base64"]) and ref(path)["sha256"] == value["sha256"]
        original_source = base64.b64decode(original_payload["variant"]["python_files"]["dec.py"]["base64"]).decode("utf-8")
        assert original_source.count(OLD) == 1
        assert (files()[0].parent / "dec.py").read_text(encoding="utf-8") == original_source.replace(OLD, NEW)
        result = tick.load(closure.artifact(ROOT, row["revision1_result"]))
        assert result["result"] == "failed" and result["original_CPU_guards_unchanged"]
        assert result["original_native_guards_unchanged"] and result["native_GPU_executed"] is False
        log = ROOT / ("pipeline/analyses/y97_rev1_cpu_" + token + ".log")
        assert not log.with_suffix(".json").exists()
        trace = log.read_text(encoding="utf-8")
        match = re.search(r'File "<stdin>", line (\d+), in capture_adapter', trace)
        assert match
        parsed = ast.parse(revised_text)
        callback = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "capture_adapter")
        failed = next(node for node in callback.body if isinstance(node, ast.Assert)
            and ast.unparse(node.test) == "torch.equal(value, projected + raw.square() @ module.weight)")
        assert int(match.group(1)) - failed.lineno == 9
        output = ROOT / ("pipeline/checks/y97_revision1_closeout_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            observations=observations, occupations=occupations, earlier_preparations=earlier,
            original_Y97_failure=row["CPU_failure"], revision1_registration=row["revision1_registration"],
            revision1_result=row["revision1_result"], original_guard_model_YAML_preserved=True,
            CPU_body_after_payload_byte_identical=True, failed_guard=ast.unparse(failed.test),
            local_program_line=failed.lineno, remote_traceback_line=int(match.group(1)),
            probe_wrapper_prefix_lines=9, component_and_anchor_control_locations_precede_failure=True,
            execution_position_is_fail_fast_inference_not_complete_CPU_PASS=True,
            complete_CPU_result_generated=False, two_synthetic_updates_PASS_established=False,
            original_Y92_Y93_and_earlier_Y94_Y95_Y96_preserved=True,
            original_full_frozen_FIRST_and_cost_preserved=True, original_GPU1_and_Y57_pause_preserved=True,
            new_scientific_candidates=0, same_ID_implementation_revisions=1, CPU_candidate_attempts=1,
            CPU_candidate_attempt_result="failed", native_GPU_tasks=0, new_admissions=0,
            new_training_launches=0, original_archives=0, original_training_checkpoints_loaded=False,
            mechanism_falsified=False, efficacy_measured=False, production_HEAD_index_queue_preserved=True,
            evidence=[ref(__file__), row["CPU_failure"], row["CPU_program"], row["component_diagnosis"],
                row["revision1_registration"], row["revision1_preparation"], row["revision1_CPU_program"],
                row["revision1_result"], ref(log)]
                + [entry["evidence"] for entry in observations.values()] + list(occupations.values())))
        row["evidence"].append(ref(output))
        assert work["tick_attempt"]["kind"] == "resource_check"
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, lines = memory(token, view, row, observations, occupations, output)
        decision = monitor.log_decision(str(ROOT), "other",
            "Y97同编号事前登记的逐token执行修订仍未通过完整CPU控制，如何处置？",
            "修订1结果记failed并按implementation数值问题挂起；保留原件及修订全部证据，不弱化原守卫、不准入、不重命名或否定假设。",
            "原CPU程序除payload外逐字一致；执行越过组件及锚点位置，在共享capture_adapter的原批量矩阵乘逐位相等守卫失败；没有完整CPU结果或两步通过证据。",
            output.relative_to(ROOT).as_posix())
        phases = []
        for cid, entry in observations.items():
            phase = entry["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30，步"
                + str(phase["step"]) + "/" + str(phase["count"]))
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅处理AdaRES_project；实际owner会话、Disabled后台调度、正式skill/experiment_first、AGENTS/goal/SERVERS及GPU1暂停记录核验后begin，先查进展并尝试已有合格派发；ready waiting为0，无新终态。",
            "运行：" + "；".join(phases) + "；原PID/source/YAML/receipt/私有runtime、350W/280W及原生步数递增核实；canonical物理占用确认两个授权GPU0均busy+reserved，未干扰GPU1或外部任务。",
            "新准备：同一Y97下先登记一次局部执行修订，再以独立rev1包保留原模型/YAML字节；将相同全token二次投影改为逐token固定形状torch.mv。原CPU程序除冻结payload外逐字不变，全部数值守卫及原生观察器未改。",
            "实际CPU检查失败：已越过组件批次置换和前面的W2锚点/输入梯度/因果旁路位置，在共享capture_adapter的原批量矩阵乘torch.equal守卫触发AssertionError。执行位置是fail-fast推断，不是完整CPU或两步更新PASS。保留完整raw日志、原失败和本次修订结果；没有native计划、留版commit、GPU测试、准入或派发。",
            "较早Y94/Y95/Y96原CPU准备、独立版本及native计划保留且不重放；Y92/Y93终态计划只读核验，未加载训练checkpoint、停止或归档。新科学ID0、同编号实现修订1、CPU完整尝试1失败、GPU原生0、准入0、启动0、归档0。",
            "七节记忆先精确备份；连续刷新训练和canonical占用并保留collector的实际resource_check，随后正式report/finish；不预先宣称收尾结果。"
        ], decisions=[
            "Y97修订1按numerical_control_failure/implementation挂起，原模型、原程序和所有原检查保持。不得计时重跑、改容差或守卫、重命名绕过或宣称假设被反证；进一步局部修订须新证据、同ID、预算内事前登记并保留全部控制。",
            "原Y92/Y93继续；Y94、Y95、Y96按顺序等待匹配owner自然30轮、精确完整有限checkpoint核验归档及实时空闲GPU0，真实native通过后再按完整冻结协议准入/安全同步/递增计数器启动。挂起不阻断独立工作，无需新增用户操作。",
            "科学goal字节、原收据、完整FIRST及同设备cost、数据/环境/mappings/seed1999-fixed30/有害子组/禁用/功耗/调度/恢复要求不变。GPU1全部使用和原GPU1冻结后评/cost继续暂停，Y57另行暂停，禁止迁移替代和server6。"
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
