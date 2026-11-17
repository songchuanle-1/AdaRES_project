"""Close new pending server2 guards while preserving both live training runs."""
import argparse
import copy
from datetime import datetime
import json
import os

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y94_deployment import Y93, Y94, contract, git
import y95_server2_pre_admission as checks
import closure
import gpu_occupation


def evidence(entry):
    return entry["path"] + " SHA" + entry["sha256"]


def save_memory(token, view, candidate, observations, occupations, output):
    path = ROOT / "pipeline/codex_memory.md"
    backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
    assert not backup.exists()
    original = path.read_bytes()
    backup.write_bytes(original)
    assert backup.read_bytes() == original
    old = original.decode("utf-8").splitlines()
    assert sum(line.startswith("## ") for line in old) == 7
    lines = []
    for line in old[:old.index("## Last Tick")]:
        if line.startswith("- Current token "):
            line = ("- Current token " + token + "; snapshot " + view["generated_at"]
                    + "; deadline " + view["deadline_at"] + "; never extend.")
        elif line.startswith("- Exact previous seven-section memory:"):
            line = ("- Exact previous seven-section memory: " + evidence(ref(backup))
                    + "; ALL history/negative evidence/bans/dedicated restoration requirements "
                    "byte-preserved in the full snapshot chain; compression never reopens work.")
        elif line.startswith("- Current AdaRES-Y93 "):
            entry = observations["AdaRES-Y93"]
            phase = entry["phase"]
            line = ("- Current AdaRES-Y93 " + phase["phase"] + str(phase["epoch"]) + "/30 step"
                    + str(phase["step"]) + "/" + str(phase["count"])
                    + "; original PID47380/source/YAML/native progress/280W verified: "
                    + evidence(entry["evidence"]))
        elif line.startswith("- Current AdaRES-Y94 "):
            entry = observations["AdaRES-Y94"]
            phase = entry["phase"]
            suffix = line[line.index(". Actual original save directory"):]
            line = ("- Current AdaRES-Y94 original PID218513/server1GPU0/350W, " + phase["phase"]
                    + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"])
                    + "; native counter increase/source/YAML verified " + evidence(entry["evidence"]) + suffix)
        elif line.startswith("- Fresh canonical occupation:"):
            line = ("- Fresh canonical occupation: "
                    + "; ".join(evidence(entry) for entry in occupations.values())
                    + "; both authorizedGPU0 busy/reserved, age<300s required; collector actual tick_attempt retained.")
        elif line.startswith("- Next tick first observeY93/Y94"):
            line = ("- Next tick first observeY93/Y94 and dispatch already admitted work. Y95 "
                    "HEADY94/current-owner binding and server2 pre-admission guards ready, native GPU "
                    "still unexecuted. Only after matching original Y93/server2 natural30/full finite "
                    "exact checkpoint/canonical archive and fresh idleGPU0 execute bound native runner; "
                    "actual PASS permits read-only y95_server2_pre_admission.py verifier. Before source "
                    "selection NEW actual-Y95-native-tested Y94 additive terminal binding is mandatory, "
                    "then exact source/full freeze/receipt/admission/safe dispatch/increasing counters "
                    "in that tick. Server1 alternative needs its own matching chain; Y96/Y98 need "
                    "NEW bindings on currentHEAD. No old CPU/version/plan/terminal replay or Y97 retry.")
        lines.append(line)
    where = next(index for index, line in enumerate(lines) if line.startswith("- Y95 ONLY current runner=")) + 1
    lines[where:where] = [
        "- NEW Y95 server2 pre-admission " + evidence(candidate["server2_pre_admission_preparation"])
        + ":17 new actual-artifact source/private-runtime/280W/current-owner/missing-native/full-original-protocol "
        "guards passed. Helpery95_server2_pre_admission.py is read-only and rejects missing ACTUAL native "
        "result or unarchived Y93. Full original Y93 training/review inherited only IN MEMORY; "
        "no native-result fixture, registered freeze, admission, source selection or complete launch toolchain.",
        "- Original Y95 code968112d44c506dcdd8dcf7709af997fb731e73b0, scientific CPU result/program/"
        "callbacks/both native plans/current server2 activation remain unchanged. No scientific CPU or "
        "version/binding tests replay; unchanged source/current-owner chain is required at real GPU execution.",
    ]
    lines += [
        "## Last Tick",
        "- " + token + ": original Y93/Y94 source/PID/native counters/YAML/receipts/private runtimes/"
        "280W-350W verified. Both authorizedGPU0 remain busy/reserved; no new terminal or safe GPU "
        "diagnostic slot. No current/old checkpoint loaded, stop, restart, migration or archive.",
        "- Earlier Y95 server2 pre-admission helper/17 NEW actual-artifact boundary tests passed "
        "first execution; missing actual native evidence/current still-running Y93/source/runtime/"
        "GPU1/power/goal/protocol substitutions refused. Complete original evaluation/training "
        "inheritance bound to actual original parent hash, IN MEMORY ONLY.",
        "- New scientificIDs0/scientificCPUattempts0/nativeGPUtasks0/admissions0/launches0/archives0; "
        "new operational group1(17 tests). Original Y95 scientific files/CPU callbacks/version/"
        "both native programs/plans/current server2 activation preserved; no fixture/native PASS, "
        "registered freeze or full launch-toolchain claim. Y96/Y98 and Y97 original/revision1 failures untouched.",
        "- Full9508x141 FIRST/two repeats/bootstrap10000 seed1999/TRAIN1487-3325/all harm metrics/"
        "original-device cost/goal bytes/receipts/data/runtime/mappings/fixed30/bans/dedicated restore "
        "rules/power/scheduler intact. GPU1 all use and Y57 separately paused; no server6/cross-project/foreign-job change.",
        "- Closeout " + evidence(ref(output))
        + "; exact memory backup, fresh sequential native observations/canonical physical occupation, "
        "collector actual resource_check preserved. Verify official finish "
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
            terminal = tick.load(closure.artifact(ROOT, candidates[cid]["terminal_handoff_plan"]))
            assert terminal["checkpoint_loaded"] is False and terminal["archived"] is False
            assert terminal["post_training_freeze"] == candidates[cid]["post_training_freeze"]
        candidate, _, _, _ = checks.pending(work, Y94)
        assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
        assert candidate["required_server"] == "server-2" and candidate["required_gpu"] == 0
        assert candidate["native_smoke_plan"] == candidate["native_smoke_plan_server2_gpu0"]
        assert not candidate.get("native_smoke_result") and not candidate.get("admission_receipt")
        assert not (ROOT / "pipeline/admissions/AdaRES-Y95.json").exists()
        preparation_ref = candidate["server2_pre_admission_preparation"]
        preparation = tick.load(closure.artifact(ROOT, preparation_ref))
        assert preparation["token"] == token and preparation["working_HEAD"] == Y94
        assert preparation["pending_operation_tests_passed"] == 17
        for key in ("native_GPU_executed", "CPU_scientific_tests_replayed", "native_result_fixture_created",
                    "formal_freeze_registered", "full_launch_toolchain_prepared", "admitted", "queued"):
            assert preparation[key] is False
        assert preparation["original_candidate_activation_preserved"]
        assert preparation["both_original_native_plans_preserved"]
        assert preparation["full_original_training_and_review_in_memory_only"]
        assert preparation["parent_freeze"] == candidates["AdaRES-Y93"]["post_training_freeze"]
        for entry in preparation["evidence"]:
            closure.artifact(ROOT, entry)
        log_ref = ref("pipeline/checks/y95_server2_pre_admission_" + token + ".log")
        raw = closure.artifact(ROOT, log_ref).read_bytes()
        assert b"Ran 17 tests" in raw and b"OK" in raw and b"skipped" not in raw
        for cid in ("AdaRES-Y96", "AdaRES-Y98"):
            row = candidates[cid]
            assert row["stage"] == "blocked" and not row.get("native_smoke_result")
            assert not row.get("admission_receipt")
            for key in ("CPU_result", "source_manifest", "native_preparation",
                        "native_smoke_plan_server1", "native_smoke_plan_server2_gpu0"):
                closure.artifact(ROOT, row[key])
        numerical = candidates["AdaRES-Y97"]
        assert numerical["stage"] == "blocked" and numerical["blocker_kind"] == "numerical_control_failure"
        assert len(numerical["revision_history"]) == 1 and numerical["revision_history"][0]["result"] == "failed"
        closure.validate_revisions(ROOT, numerical)
        output = ROOT / ("pipeline/checks/y95_server2_pre_admission_closeout_" + token + ".json")
        assert not output.exists()
        dependencies = [ref(__file__), preparation_ref, log_ref, numerical["CPU_failure"],
            numerical["revision1_result"]] + [
                entry["evidence"] for entry in observations.values()] + list(occupations.values())
        tick.save(output, dict(
            complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            Y95_new_pending_operation_tests=17, Y95_pre_admission_preparation=preparation_ref,
            full_original_protocol_in_memory_only=True, native_result_fixture_created=False,
            original_Y93_Y94_PID_training_source_yaml_receipts_full_freezes_preserved=True,
            original_Y95_CPU_callbacks_source_native_plans_and_activation_preserved=True,
            original_Y96_Y98_and_Y97_failures_preserved=True,
            scientific_CPU_programs_or_completed_tests_replayed=False,
            original_training_checkpoints_loaded=False, formal_freeze_registered=False,
            full_launch_toolchain_prepared=False,
            new_scientific_candidates=0, scientific_CPU_candidate_attempts=0, native_GPU_tasks=0,
            new_admissions=0, new_training_launches=0, exact_original_archives=0,
            production_HEAD_index_queue_preserved=True, full_original_FIRST_and_cost_preserved=True,
            original_GPU1_and_Y57_separate_pause_preserved=True,
            observations=observations, occupations=occupations, efficacy_measured=False, evidence=dependencies))
        attempt = copy.deepcopy(work["tick_attempt"])
        assert attempt["kind"] == "resource_check"
        candidate["evidence"].append(ref(output))
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, lines = save_memory(token, view, candidate, observations, occupations, output)
        decision = monitor.log_decision(str(ROOT), "other",
            "已有Y95执行绑定就绪但授权GPU0均忙，下一步可安全准备什么？",
            "优先为Y95准备server2准入前只读检查，17项新测试首轮通过，保持当前server2激活和双原生计划不变。",
            "真实native结果缺失、Y93未完整结束归档、错误源/HEAD/goal/私有runtime/设备/280W/"
            "协议等均拒绝。原Y93完整训练与冻结评测仅内存继承，不注册freeze，不造native "
            "fixture，不宣称GPU原生、完整启动工具链、准入或科学收益。",
            output.relative_to(ROOT).as_posix())
        phases = []
        for cid, entry in observations.items():
            phase = entry["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30，步"
                          + str(phase["step"]) + "/" + str(phase["count"]))
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅AdaRES_project；实际owner、后台Disabled、正式skill/experiment_first、AGENTS/goal/"
            "SERVERS及GPU1暂停核验。先查询进展及派发现有合格队列，本轮ready waiting为0，无新终态。",
            "运行：" + "；".join(phases) + "。原PID/source/YAML/receipt/私有runtime/"
            "精确数据、350W/280W及原生步数增长确认；两授权GPU0 canonical只读占用均busy+reserved。"
            "不抢占当前训练或GPU1外部任务，不加载live checkpoint，不停止或归档运行。",
            "新准备：较早Y95的server2准入前边界检查17项首次通过，使用当前同字节七文件/"
            "HEADY94绑定/双原生payload、当前owner及原Y93完整冻结训练评测协议。缺失实际native "
            "结果、Y93未完整结束归档、错误source/HEAD/goal/private runtime/GPU1/设备/280W/协议均拒绝。",
            "完整原训练及9508表达141场景/candidate+Y36-W2-B1各两次/FIRST主报告/"
            "场景bootstrap10000-seed1999/TRAIN1487-3325/全部有害子组与指标/原设备cost协议"
            "绑定原哈希工件，仅内存继承测试，不注册Y95冻结计划、准入收据或原生结果fixture。",
            "Y95科学源/CPU结果程序/native callbacks/当前版本/双原生程序与计划/当前server2 "
            "激活不变。Y96/Y98与Y97原始及修订1数值失败保持；已完成CPU/version/binding/"
            "原生及历史终态不重放。Y93/Y94已有终态计划仅核验工件，未执行完整checkpoint核验。",
            "本轮新科学ID0、科学CPU尝试0、GPU原生0、准入0、启动0、归档0；新操作测试组1计17项。"
            "精确备份七节记忆、刷新训练和canonical物理占用、保留collector实际resource_check，"
            "再正式report/finish，不预称收尾PASS。"
        ], decisions=[
            "先推进较早Y95，不为凑额度再造或复制机制。只有HEADY94、匹配原Y93/server2 "
            "自然30轮完整有限checkpoint核验及canonical归档、实时独占空闲GPU0后执行原"
            "run_y95_bound_native_y94.py --server server-2。实际native通过后可用新只读verifier；"
            "源选择前仍须绑定实际Y95原生证据的新Y94终态加法兼容测试，继而当轮精确源选择/"
            "完整freeze/receipt/准入/canonical派发/递增counter。server1替代路径须自身匹配执行链。",
            "科学goal字节、原签名收据、原数据/private runtime/mappings/seed1999固定30轮/"
            "禁用及专用恢复/功耗/调度/完整FIRST和原设备cost不变。GPU1全部使用及原GPU1 "
            "冻结后评继续暂停，Y57单独暂停；不迁移、不替代资源、不用server6、不跨项目、不干预外部任务。"
            "无需新用户操作。"
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
