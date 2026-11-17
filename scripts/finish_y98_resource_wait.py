"""Verify continuing original Y96/Y98 training and close actual resource wait."""
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
        assert not view["preflight"]["operational_issues"] and not view["changes"]["new_terminal_ids"]
        assert view["queue"]["ready_waiting_count"] == 0
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
        y99 = candidates["AdaRES-Y99"]
        assert y99["stage"] == "blocked" and y99["blocker_kind"] == "exclusive_gpu_slot"
        assert y99["code_commit"] == "53f5b028bdf24e8b41dd934931d1fedc99843d8d"
        assert not y99.get("native_smoke_result") and not y99.get("admission_receipt")
        binding = tick.load(closure.artifact(ROOT, y99["additive_version_binding"]))
        assert binding["parent_commit"] == Y98 and binding["guard_tests_passed"] == 10
        assert binding["matching_current_owners"] == {"server-1": "AdaRES-Y96", "server-2": "AdaRES-Y98"}
        attempt = work["tick_attempt"]
        assert attempt["tick_started_at"] == view["generated_at"]
        assert attempt["kind"] == "resource_check" and attempt["result"] == "blocked"
        output = ROOT / ("pipeline/checks/y98_resource_wait_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            observations=observations, occupations=occupations, original_runs_preserved=True,
            new_scientific_candidates=0, scientific_CPU_replays=0, native_GPU_tasks=0,
            admissions=0, launches=0, archives=0, checkpoints_loaded=False,
            repeated_scientific_or_binding_tests=False, formal_FIRST_or_cost_executed=False,
            evidence=[ref(__file__), y99["additive_version_binding"]]
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
            elif line.startswith("- AdaRES-Y96 original PID") or line.startswith("- AdaRES-Y98 original PID"):
                cid = line.split()[1]
                row = observations[cid]
                phase = row["phase"]
                name, _, pid, commit = next(r for r in RUNS if r[1] == cid)
                line = "- " + cid + " original PID" + str(pid) + "/" + name + "GPU0, " + phase["phase"] + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"]) + "; native counters/source/YAML/private runtime/power verified " + evidence(row["evidence"]) + "; commit" + commit + "."
                updated.add(cid)
            elif line.startswith("- Fresh canonical occupation "):
                line = "- Fresh canonical occupation " + "; ".join(evidence(r) for r in occupations.values()) + "; both authorizedGPU0 busy/reserved; collector actual tick_attempt retained. Original GPU1 frozen FIRST/cost remain held, never moved/substituted."
                updated.add("occupation")
            lines.append(line)
        assert updated == {"token", "backup", "AdaRES-Y96", "AdaRES-Y98", "occupation"}
        lines += ["## Last Tick",
            "- " + token + ": explicit user continue-training request. Canonical begin observed/dispatched existing qualified work first; ready waiting0/no terminal/no operational issues. Original Y96/Y98 native counters increase, exact PID/source/YAML/receipt/private runtime/350W280W verified; both authorizedGPU0 fresh busy/reserved.",
            "- Y99 same-byte HEADY98/current-owner binding and both exact original terminal plans retained without repeated tests. New scientificIDs0/scientificCPUreplays0/nativeGPUtasks0/admissions0/launches0/archives0/FIRST0/cost0. No stop/restart/migration/checkpoint loading/terminal replay or new user action. Goal/receipts/data/runtime/seed1999fixed30/bans/GPU1+Y57 pauses intact. Closeout " + evidence(ref(output)) + "; official finish/receipt/release authoritative."]
        assert len(lines) < 120 and sum(r.startswith("## ") for r in lines) == 7
        memory.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        phases = [cid + " " + r["phase"]["phase"] + str(r["phase"]["epoch"]) + "/30 step" + str(r["phase"]["step"]) + "/" + str(r["phase"]["count"]) for cid, r in observations.items()]
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "依用户继续训练指令，仅处理AdaRES_project；已核验实际owner、后台Disabled和当前授权后begin，先查进展并尝试派发已有合格任务，ready waiting0。",
            "运行：" + "；".join(phases) + "。原PID、源码、YAML、签名收据、私有运行环境、350W/280W和原生步数递增已核验；无新终态或操作故障。",
            "两张授权GPU0当前物理busy且保留给原运行；canonical只读资源检查已留档，保留collector真实resource_check。本轮无新原生检查、准入、启动、归档或正式评测。",
            "已有Y99当前HEAD/原运行资源绑定和Y96/Y98精确终态工具保持，不重跑CPU/版本/绑定/原生计划测试，不新增占位候选或重试已冻结失败路线。"
        ], decisions=[
            "保留Y96/Y98原训练继续至固定30轮。匹配原运行自然结束、精确完整稳定有限last/best/model/optimizer/scheduler及Train/Val1..30核验和规范归档后，再以新鲜独占空闲未保留的授权GPU0推进Y99真实原生YAML检查；通过后同轮执行完整原设备协议冻结、准入、安全部署和步数验证。",
            "全部server2GPU1使用和Y57单独暂停继续保留；不迁移冻结后评或成本、不重启、不使用server6、不跨项目、不改调度或功耗、不干预外部任务。科学goal字节、旧收据、原数据/运行环境/映射/seed1999固定30轮/禁用项/专用恢复要求保持，无新增用户处理事项。"
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
        observations=observations_for_output(output), native_GPU_tasks=0, admissions=0, launches=0, archives=0)))


def observations_for_output(output):
    return {cid: row["phase"] for cid, row in tick.load(output)["observations"].items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    finish(parser.parse_args().token)
