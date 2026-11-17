"""Close operational preparation without replaying science or occupying live GPUs."""
import argparse
import ast
from datetime import datetime
import json

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y95_deployment import Y94, Y95, contract, git
from run_y96_bound_native_y95 import validate_binding, runtime
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


def evidence(entry):
    return entry["path"] + " SHA" + entry["sha256"]


def memory(token, view, candidates, observations, occupations, output):
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
        elif line.startswith("- Y95 current original server2GPU0/280W PID"):
            row = observations["AdaRES-Y95"]
            phase = row["phase"]
            suffix = line[line.index(". Exact tested source "):]
            line = ("- Y95 current original server2GPU0/280W PID34484, " + phase["phase"]
                    + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"])
                    + " native counters increasing: " + evidence(row["evidence"]) + suffix)
        elif line.startswith("- Current AdaRES-Y94 "):
            row = observations["AdaRES-Y94"]
            phase = row["phase"]
            suffix = line[line.index(". Original exact directory"):]
            line = ("- Current AdaRES-Y94 original PID218513/server1GPU0/350W, " + phase["phase"]
                    + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"])
                    + "; native counter/source/YAML verified " + evidence(row["evidence"]) + suffix)
        elif line.startswith("- Fresh canonical occupation:"):
            line = ("- Fresh canonical occupation: "
                    + "; ".join(evidence(entry) for entry in occupations.values())
                    + "; both authorizedGPU0 busy/reserved; age<300s required; collector actual tick_attempt retained.")
        elif line.startswith("- Next tick first observeY94/Y95"):
            line = ("- Next tick first observeY94/Y95 using y95_deployment.py and dispatch admitted work. "
                    "Y93 archive must NOT be replayed. Y96 now has a NEW tested same-byte HEADY95/current-owner "
                    "binding; only after matching Y94/server1 orY95/server2 naturally completes full30, exact stable "
                    "finite original checkpoint verification/canonical archive and fresh exclusive idle/unreservedGPU0 "
                    "use run_y96_bound_native_y95.py. Actual native PASS then NEW actual-native-tested terminal "
                    "compatibility for the other live owner, selected-device full original freeze/receipt/admission/"
                    "exact source/safe dispatch/increasing counters in that tick. Y98/Y99 still need NEW bindings; "
                    "Y96 thenY98 thenY99, no filler or Y97 retry.")
        lines.append(line)
    candidate = candidates["AdaRES-Y96"]
    where = lines.index("## Active Hypotheses") + 1
    lines[where:where] = [
        "- Y96 CURRENT identical seven CPU-tested additions retained on actual-nativeY95: commit"
        + candidate["code_commit"] + "/" + candidate["retained_branch"] + "; NEW binding "
        + evidence(candidate["additive_version_binding"]) + ",10 operational guards passed. Original "
        "commit994db457a3b00444b6ace4f0c9b7f3b6e047c83e/branch/source/CPU/callback/native plans retained "
        "in original_pre_y95_binding and unchanged artifacts; historical pre-binding entries below are NOT current execution authority.",
        "- Y96 ONLY current runner=run_y96_bound_native_y95.py at HEADY95/currentY94-server1 orY95-server2 "
        "after natural30/full exact finite original checkpoint/archive plus fresh exclusive idle/unreservedGPU0. "
        "Both scientific native program bodies and frozen runtime/data payloads identical; only code_commit metadata "
        "changed. No scientific CPU replay/native GPU/result/admission/launch; later HEAD/owner change needs NEW binding.",
    ]
    where = lines.index("## Open Questions And Risks")
    lines[where:where] = [
        "- NEW Y95 original exact terminal preparation " + evidence(candidates["AdaRES-Y95"]["terminal_handoff_plan"])
        + ";29 lifecycle guards passed. Actual native log declares original save directory20261109_081912, NOT "
        "launch timestamp20261109_212023. Future finalize_y95_fixed30.py ONLY at HEADY95; no checkpoint "
        "loaded/stopped/archived. Original server2 private runtime/exact data/280W/source/YAML/freeze/receipt "
        "and all original helpers preserved; future HEAD changes require NEW tested additive binding."
    ]
    lines += [
        "## Last Tick",
        "- " + token + ": original Y94/Y95 source/YAML/PID/receipts/private runtime/power/native counter "
        "increase verified. Both authorizedGPU0 busy/reserved; no new terminal or operational failure.",
        "- FIRST Y95 exact original terminal helpers and declared-location discovery prepared read-only,29 lifecycle "
        "guards passed. No checkpoint loaded, training stopped, restart/migration/archive or GPU test.",
        "- NEW Y96 same-byte binding onto actual-nativeY95/currentY94-server1/Y95-server2 passed10 guards; "
        "original CPU/results/programs/callbacks/native bodies/negative evidence preserved. Main HEAD/index/"
        "working files/queue/current runs unchanged; no filler/scientific retry or weakened control.",
        "- New scientificIDs0/scientificCPUattempts0/nativeGPUtasks0/admissions0/launches0/archives0; "
        "no formal FIRST/cost/promotion. Y98/Y99 old prepared artifacts and Y97 original/revision1 failures retained. "
        "All GPU1 use and Y57 separately paused; goal bytes/old receipts/data/protocol/runtime/mappings/bans/"
        "dedicated recovery/350W-280W/scheduler/foreign jobs intact.",
        "- Closeout " + evidence(ref(output))
        + "; exact memory backup and collector actual resource_check retained. Official finish must confirm "
        "FINALIZED/PASS/context_committed/released/absent owned lease/pending archive journal.",
    ]
    assert len(lines) < 120 and sum(line.startswith("## ") for line in lines) == 7
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return ref(backup), len(lines)


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y95
        before = {path: ref(path) for path in
                  ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        assert before["pipeline/goal.json"]["sha256"] == GOAL
        assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == PAUSE
        for cid, digest in (
            ("AdaRES-Y93", "f6b87f2813f1e8e429772bc0f63963fe1a43e0582c6c65e21771092dcbf69832"),
            ("AdaRES-Y94", "b9deb51efa3c274283f8a23ae678d42a4f0c11a29e758d19ae696e80df13cb84"),
            ("AdaRES-Y95", "de4e2620341bcd651a6d2ad3d8bfa98dc7e2e8523a7da9f1ffaea8c11d7597c5"),
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
        servers = monitor.load_servers(str(ROOT))
        assert servers["server-1"]["allowed_gpus"] == servers["server-2"]["allowed_gpus"] == [0]
        assert servers["server-6"]["allowed_gpus"] == []
        observations, occupations = {}, {}
        for name, cid, pid, commit in (
            ("server-1", "AdaRES-Y94", 218513, Y94), ("server-2", "AdaRES-Y95", 34484, Y95),
        ):
            task = next(row for row in queue["gpu_queues"][name] if row["id"] == cid)
            assert (task["status"], task["launch_verified"], task["pid"], task["code_commit"],
                    str(task["gpu"])) == ("running", True, pid, commit, "0")
            closure.admission(ROOT, task)
            contract(commit, Y95, cid)
            entry = candidates[cid]["latest_training_observation"]
            value = tick.load(closure.artifact(ROOT, entry))
            assert value["token"] == token and value["native_progress_confirmed"] is True
            assert value["pid"] == pid and value["source_marker"] in (commit, Y95)
            fresh(value, "observed_local")
            observations[cid] = dict(evidence=entry, phase=value["after"]["latest"], pid=pid)
            proof_ref = ref("pipeline/checks/gpu_occupation_" + name + "_release_" + token + ".json")
            fresh(tick.load(closure.artifact(ROOT, proof_ref)), "observed_at")
            held = next(row for row in work["candidates"] if row.get("stage") == "blocked"
                and row.get("blocker_kind") == "exclusive_gpu_slot" and row.get("required_server") == name)
            server = next(row for row in context["queue"]["servers"] if row["name"] == name)
            state = gpu_occupation.validated_state(ROOT, context, held, server, queue)
            assert state["busy_gpu_ids"] == state["reserved_gpu_ids"] == [0] and not state["idle_gpu_ids"]
            occupations[name] = proof_ref
        terminal_ref = candidates["AdaRES-Y95"]["terminal_handoff_plan"]
        terminal = tick.load(closure.artifact(ROOT, terminal_ref))
        assert terminal["token"] == token and terminal["CPU_guard_tests_passed"] == 29
        assert terminal["checkpoint_loaded"] is False and terminal["archived"] is False
        assert terminal["admission"] == ref("pipeline/admissions/AdaRES-Y95.json")
        assert terminal["post_training_freeze"] == candidates["AdaRES-Y95"]["post_training_freeze"]
        location = tick.load(closure.artifact(ROOT, terminal["declared_location"]))
        assert location["checkpoint_directory"] == "20261109_081912"
        for entry in terminal["dependencies"]:
            closure.artifact(ROOT, entry)
        candidate = candidates["AdaRES-Y96"]
        assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
        assert not candidate.get("native_smoke_result") and not candidate.get("admission_receipt")
        binding_ref = candidate["additive_version_binding"]
        binding = tick.load(closure.artifact(ROOT, binding_ref))
        validate_binding(binding, candidate, Y95)
        assert binding["token"] == token and binding["guard_tests_passed"] == 10
        assert binding["scientific_revision"] is False and binding["native_executed"] is False
        assert git("rev-parse", candidate["retained_branch"]).decode().strip() == candidate["code_commit"]
        for entry in binding["evidence"]:
            closure.artifact(ROOT, entry)
        for name, key in (("server-1", "native_smoke_plan_server1"),
                          ("server-2", "native_smoke_plan_server2_gpu0")):
            plan = tick.load(closure.artifact(ROOT, candidate[key]))
            program = closure.artifact(ROOT, plan["program"])
            frozen = ast.literal_eval(ast.parse(program.read_text(encoding="utf-8")).body[0].value)
            assert validate_frozen(candidate, plan, frozen) == 7
            assert runtime(name, candidate, frozen, servers[name]) == (350. if name == "server-1" else 280.)
        output = ROOT / ("pipeline/checks/y96_y95_binding_closeout_" + token + ".json")
        assert not output.exists()
        dependencies = [ref(__file__), terminal_ref, binding_ref,
            ref("pipeline/checks/y95_terminal_helpers_" + token + ".json"),
            ref("pipeline/checks/y95_terminal_wrapper_" + token + ".py"),
            ref("pipeline/checks/y96_y95_binding_wrapper_" + token + ".py")] + [
                row["evidence"] for row in observations.values()] + list(occupations.values())
        tick.save(output, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            original_Y95_terminal_preparation_guards=29, Y96_operational_binding_guards=10,
            only_native_code_commit_metadata_changed=True, original_native_program_bodies_identical=True,
            original_running_checkpoints_loaded=False, scientific_CPU_attempts=0,
            new_scientific_candidates=0, native_GPU_tasks=0, admissions=0, launches=0, archives=0,
            production_HEAD_index_queue_preserved=True, full_original_protocol_and_pauses_preserved=True,
            scientific_efficacy_measured=False, observations=observations, occupations=occupations,
            evidence=dependencies))
        candidate["evidence"].append(ref(output))
        candidates["AdaRES-Y95"]["evidence"].append(ref(output))
        candidates["AdaRES-Y95"]["next_action"] = (
            "Observe original PID34484 native progress. Preserve exact original save directory20261109_081912. "
            "Only after natural fixed30 and process absence use finalize_y95_fixed30.py at HEADY95 for full "
            "stable finite original last/best/model/optimizer/scheduler/log/source/YAML/receipt verification "
            "and canonical archive. A changed HEAD requires a NEW tested additive binding. Do not restart.")
        assert work["tick_attempt"]["kind"] == "resource_check"
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, lines = memory(token, view, candidates, observations, occupations, output)
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅AdaRES_project。核对实际owner、Disabled后台调度及正式前台/experiment_first规则、"
            "AGENTS/goal/SERVERS/最新GPU1暂停后begin；先核运行及派发现有合格任务，ready waiting为0。"
            "快照列出的Y93是上轮已完整核验归档的终态，本轮未重做checkpoint核验或归档。",
            "原Y94 PID218513/server-1 GPU0/350W和原Y95 PID34484/server-2 GPU0/280W的source、YAML、"
            "收据、私有runtime及原生计数递增成立。canonical只读物理查询确认两张授权GPU0均busy+reserved，"
            "无可独占空闲卡；未抢占、改功耗或使用GPU1。",
            "新准备：Y95专用精确原终态工具，实际日志声明原保存目录20261109_081912，"
            "不是启动时间20261109_212023。29项生命周期guard通过，原收据/freeze/源和全部旧工具保留；"
            "未加载运行中的checkpoint、停止/重启/迁移训练或提前归档。",
            "推进已有Y96：相同七个CPU已测新增文件绑定到实际native通过的HEADY95，"
            "匹配当前Y94/server1和Y95/server2；10项新操作guard通过。两份原native程序体、"
            "CPU结果/程序、回调和冻结科学/runtime/data内容不变，仅code_commit元数据更新；"
            "主HEAD/index/queue和其他候选保持。未重放CPU科学测试或GPU原生检查。",
            "本轮新科学ID0、CPU科学尝试0、GPU原生0、准入0、启动0、归档0、正式评测0。"
            "原Y98/Y99准备和Y97原始/修订1失败保持。精确备份七节记忆，保留collector真实resource_check，"
            "以官方report/finish收据和租约释放为准，不预签收尾结果。"
        ], decisions=[
            "既有三个CPU准备候选优先，不新增占位变体或重放已通过的科学检查。Y96先行；"
            "仅在当前HEADY95、匹配原运行自然fixed30完整checkpoint核验/canonical归档、"
            "实时独占空闲且未保留的授权GPU0上执行run_y96_bound_native_y95.py。"
            "实际native通过再建立另一在跑任务的新测试终态兼容绑定，并当轮完成设备匹配的完整"
            "原freeze/receipt/准入/精确source/安全派发/递增步数。未来HEAD/owner变化须新绑定。",
            "本轮仅运行保护和待跑操作准备，无新增效果结果或用户行动项。保留科学goal字节、"
            "全部原签名收据、精确数据/私有runtime/部署映射/seed1999固定30轮和完整"
            "FIRST/有害子组/原设备成本协议。全部GPU1使用及原GPU1冻结后评继续暂停，Y57独立暂停；"
            "不跨项目、不用server6、不改调度、不干预外部任务。"
        ], evidence=[ref(output), backup]))
        assert all(ref(path) == entry for path, entry in before.items())
        assert git("ls-files", "--stage", "-z") == index
        return output, notes, lines
    finally:
        writer.close()


def finish(token):
    lease(token, 240)
    for cid in ("AdaRES-Y94", "AdaRES-Y95"):
        command(["scripts/y95_deployment.py", "observe", "--id", cid,
                 "--token", token, "--label", "release"])
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
        archive_pending_files=0, new_candidates=0, native_GPU_tasks=0, admissions=0, launches=0, archives=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    finish(parser.parse_args().token)
