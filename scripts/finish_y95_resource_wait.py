"""Close unchanged Y94/Y95 resource waits without replaying scientific checks."""
import argparse
from datetime import datetime
import json

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y95_deployment import Y94, Y95, git
import closure
import gpu_occupation


RUNS = (("server-1", "AdaRES-Y94", 218513, Y94),
        ("server-2", "AdaRES-Y95", 34484, Y95))


def evidence(entry):
    return entry["path"] + " SHA" + entry["sha256"]


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
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        assert running == {"server-1": {"0": "AdaRES-Y94"}, "server-2": {"0": "AdaRES-Y95"}}
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {row["id"]: row for row in work["candidates"]}
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        context = tick.load(ROOT / "pipeline/headless/.tick_context.json")
        assert not view["changes"]["new_terminal_ids"] and not view["preflight"]["operational_issues"]
        assert view["queue"]["ready_waiting_count"] == 0
        paused = next(row for row in queue["gpu_queues"]["server-2"] if row["id"] == "AdaRES-Y57")
        assert paused["status"] == "paused" and paused["dispatch_hold"]
        assert paused["resume_requires_user_authorization"] is True
        observations, occupations = {}, {}
        for name, cid, pid, commit in RUNS:
            task = next(row for row in queue["gpu_queues"][name] if row["id"] == cid)
            assert (task["status"], task["launch_verified"], task["pid"], task["code_commit"],
                    str(task["gpu"])) == ("running", True, pid, commit, "0")
            closure.admission(ROOT, task)
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
        for cid in ("AdaRES-Y96", "AdaRES-Y98", "AdaRES-Y99"):
            candidate = candidates[cid]
            assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
            assert not candidate.get("native_smoke_result") and not candidate.get("admission_receipt")
        assert candidates["AdaRES-Y96"]["code_commit"] == "acb8c1769b7f3d66a92d287da8859a06a138b8b0"
        attempt = work["tick_attempt"]
        assert attempt["tick_started_at"] == view["generated_at"] and attempt["kind"] == "resource_check"
        assert attempt["result"] == "blocked"
        output = ROOT / ("pipeline/checks/y95_resource_wait_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            original_PIDs_source_YAML_receipts_and_native_counters_verified=True,
            new_scientific_candidates=0, scientific_CPU_attempts=0, native_GPU_tasks=0,
            admissions=0, launches=0, archives=0, checkpoints_loaded=False,
            repeated_scientific_or_binding_tests=False, scientific_efficacy_measured=False,
            observations=observations, occupations=occupations,
            evidence=[ref(__file__)] + [row["evidence"] for row in observations.values()]
                + list(occupations.values())))
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
                line = ("- Current token " + token + "; snapshot " + view["generated_at"]
                        + "; deadline " + view["deadline_at"] + "; never extend.")
                updated.add("token")
            elif line.startswith("- Exact previous seven-section memory:"):
                line = ("- Exact previous seven-section memory: " + evidence(ref(backup))
                        + "; all history/bans/negative evidence/dedicated recovery byte-preserved; never reopening authority.")
                updated.add("backup")
            elif line.startswith("- Y95 current original server2GPU0/280W PID"):
                row = observations["AdaRES-Y95"]
                phase = row["phase"]
                suffix = line[line.index(". Exact tested source "):]
                line = ("- Y95 current original server2GPU0/280W PID34484, " + phase["phase"]
                        + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"])
                        + " native counters increasing: " + evidence(row["evidence"]) + suffix)
                updated.add("Y95")
            elif line.startswith("- Current AdaRES-Y94 "):
                row = observations["AdaRES-Y94"]
                phase = row["phase"]
                suffix = line[line.index(". Original exact directory"):]
                line = ("- Current AdaRES-Y94 original PID218513/server1GPU0/350W, " + phase["phase"]
                        + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"])
                        + "; native counter/source/YAML verified " + evidence(row["evidence"]) + suffix)
                updated.add("Y94")
            elif line.startswith("- Fresh canonical occupation:"):
                line = ("- Fresh canonical occupation: "
                        + "; ".join(evidence(entry) for entry in occupations.values())
                        + "; both authorizedGPU0 busy/reserved; age<300s required; collector actual tick_attempt retained.")
                updated.add("occupation")
            lines.append(line)
        assert updated == {"token", "backup", "Y94", "Y95", "occupation"}
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        lines += [
            "## Last Tick",
            "- " + token + ": begin/design checked progress and attempted admitted work first; ready waiting0, "
            "no new terminal/operational failure. Original Y94/Y95 PID/source/YAML/receipts/private runtime/"
            "350W-280W/native counter increase verified; fresh canonical both authorizedGPU0 busy/reserved.",
            "- Three CPU-prepared variants Y96/Y98/Y99 await native resources; Y96 current HEADY95/current-owner "
            "binding and exact original Y94/Y95 terminal plans retained. No repeated CPU/version/binding/control/"
            "native-plan tests, filler candidate or Y97 original/revision1 retry.",
            "- New scientificIDs0/scientificCPUattempts0/nativeGPUtasks0/admissions0/launches0/archives0; no "
            "checkpoint loading/terminal replay/stop/restart/migration/formal FIRST/cost/promotion. All GPU1 use "
            "and Y57 separately paused; original goal/receipts/data/full protocol/runtime/mappings/bans/"
            "dedicated recovery/power/scheduler/foreign jobs intact.",
            "- Fresh resource closeout " + evidence(ref(output)) + "; exact memory backup and collector actual "
            "resource_check retained. Notes " + notes.relative_to(ROOT).as_posix()
            + "; official finish must confirm FINALIZED/PASS/context_committed/released/no owned lease or pending archive journal.",
        ]
        assert len(lines) < 120 and sum(line.startswith("## ") for line in lines) == 7
        memory.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        phases = []
        for cid, row in observations.items():
            phase = row["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30 step"
                          + str(phase["step"]) + "/" + str(phase["count"]))
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅AdaRES_project，核验实际owner、Disabled后台调度、正式前台/experiment_first及"
            "AGENTS/goal/SERVERS/最新GPU1暂停后begin；先查进展并派发已有合格任务，ready waiting为0。",
            "运行：" + "；".join(phases) + "。原PID/source/YAML/签名收据、私有runtime、"
            "350W/280W和原生计数递增核验成立；无新终态或操作故障，未停止、重启或迁移。",
            "canonical只读物理GPU/compute PIDs和canonical reservation确认两张授权GPU0均busy+reserved，"
            "无独占空闲卡。保留collector真实resource_check，未执行GPU任务或将资源复查当科学PASS。",
            "保留已有Y96/Y98/Y99准备，优先当前HEADY95/当前owner已绑定的Y96；Y94/Y95专用精确终态计划、"
            "所有旧CPU/版本/原生程序体和Y97原始/修订1负证据保持。没有重复科学或绑定测试、新增占位候选。",
            "本轮新科学ID/CPU科学尝试/GPU原生/准入/启动/归档/正式评测均为0。"
            "精确备份七节记忆，保存真实资源和训练证据后官方report/finish，以实际收据和租约释放为准。"
        ], decisions=[
            "无新科学决策或用户行动项。原训练占用全部授权GPU0时保持已有候选优先，不抢占、"
            "不凑额度设计/重测或伪造waiting；健康推进和不可行动的资源等待保持静默。",
            "只有匹配原运行自然fixed30结束、完整稳定有限last/best/model/optimizer/scheduler和"
            "Train/Val1..30精确核验/canonical归档后，现场独占空闲未保留的授权GPU0才能执行"
            "run_y96_bound_native_y95.py。Y94原目录20261109_111838、Y95原目录20261109_081912，"
            "日志结束不是完整checkpoint核验；不重放旧已归档终态。实际native通过再完成另一在跑"
            "任务的新测试终态绑定、设备匹配完整原freeze/receipt/准入/精确source/安全派发/递增计数。",
            "全部GPU1使用及原GPU1冻结后评/成本继续暂停，Y57独立暂停；不迁移替换资源、"
            "不用server6、不跨项目、不改调度或功耗、不干预外部任务。保留科学goal字节、"
            "旧签名收据、原数据/私有runtime/部署映射/seed1999固定30轮及完整FIRST/有害子组/原设备成本协议。"
        ], evidence=[ref(output), ref(backup)]))
        assert all(ref(path) == entry for path, entry in before.items())
        assert git("ls-files", "--stage", "-z") == index
        return output, notes, len(lines)
    finally:
        writer.close()


def finish(token):
    lease(token, 240)
    for _, cid, _, _ in RUNS:
        command(["scripts/y95_deployment.py", "observe", "--id", cid,
                 "--token", token, "--label", "release"])
    for name, _, _, _ in RUNS:
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
