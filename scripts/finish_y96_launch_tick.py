"""Refresh the two original runs and close the actual Y96 admission tick."""
import argparse
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y96_deployment import Y95, Y96, git
import closure
import gpu_occupation

RUNS = (("server-1", "AdaRES-Y96", 3893460, Y96),
        ("server-2", "AdaRES-Y95", 34484, Y95))


def evidence(entry):
    return entry["path"] + " SHA" + entry["sha256"]


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y96
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        assert before["pipeline/goal.json"]["sha256"] == GOAL
        assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == PAUSE
        queue = monitor.load_queue(str(ROOT))
        assert monitor.load_running(str(ROOT)) == {
            "server-1": {"0": "AdaRES-Y96"}, "server-2": {"0": "AdaRES-Y95"}}
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {row["id"]: row for row in work["candidates"]}
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        context = tick.load(ROOT / "pipeline/headless/.tick_context.json")
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
            assert value["pid"] == pid and value["source_marker"] in (commit, Y96)
            fresh(value, "observed_local")
            observations[cid] = dict(evidence=entry, phase=value["after"]["latest"], pid=pid)
            occupation = ref("pipeline/checks/gpu_occupation_" + name + "_release_" + token + ".json")
            fresh(tick.load(closure.artifact(ROOT, occupation)), "observed_at")
            held = next(row for row in work["candidates"] if row.get("stage") == "blocked"
                and row.get("blocker_kind") == "exclusive_gpu_slot" and row.get("required_server") == name)
            server = next(row for row in context["queue"]["servers"] if row["name"] == name)
            state = gpu_occupation.validated_state(ROOT, context, held, server, queue)
            assert state["busy_gpu_ids"] == state["reserved_gpu_ids"] == [0] and not state["idle_gpu_ids"]
            occupations[name] = occupation
        y94, y95, y96 = (candidates["AdaRES-" + name] for name in ("Y94", "Y95", "Y96"))
        assert y94["stage"] == "blocked" and y94["blocker_kind"] == "exclusive_gpu_slot"
        assert y94["scientific_status"] == "fixed30_complete_frozen_review_pending"
        assert y95["stage"] == y96["stage"] == "training"
        assert y96["native_smoke_result"]["path"].endswith(
            "y96_server1_smoke_411304d6e92840b8844bb9ed860554fb.json")
        raw = tick.load(closure.artifact(ROOT, y96["native_smoke_result"]))
        assert raw["losses"] == [4.615510940551758, 6.309134006500244]
        receipt = tick.load(closure.artifact(ROOT, ref(y96["admission_receipt"])))
        smoke = tick.load(closure.artifact(ROOT, receipt["native_smoke"]))
        assert smoke["command"].endswith("411304d6e92840b8844bb9ed860554fb")
        assert smoke["native_execution_token"] == "411304d6e92840b8844bb9ed860554fb"
        correction = tick.load(closure.artifact(ROOT, y96["provenance_correction"]))
        assert correction["native_reexecuted"] is correction["runtime_mutations"] is False
        closure.artifact(ROOT, correction["original_receipt"])
        closure.artifact(ROOT, correction["original_smoke"])
        handoff = tick.load(closure.artifact(ROOT, y95["terminal_handoff_plan"]))
        assert handoff["tested_additive_native_evidence"] == y96["native_smoke_result"]
        assert handoff["payload"]["allowed_markers"] == [Y95, Y96]
        assert handoff["CPU_guard_tests_passed"] == 29 and handoff["additive_guard_tests_passed"] == 8
        guardlog = ROOT / ("pipeline/checks/adares-y96_admission_guards_" + token + ".log")
        assert "Ran 14 tests" in guardlog.read_text(encoding="utf-8")
        selection = ref("pipeline/checks/adares-y96_source_selection_" + token + ".json")
        dispatch = ref("pipeline/checks/adares-y96_dispatch_" + token + ".json")
        for cid in ("AdaRES-Y98", "AdaRES-Y99"):
            assert candidates[cid]["stage"] == "blocked"
            assert not candidates[cid].get("native_smoke_result")
        output = ROOT / ("pipeline/checks/y96_launch_closeout_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            current_original_runs_verified=True, observations=observations, occupations=occupations,
            actual_native_result_reused=y96["native_smoke_result"], native_GPU_tasks_this_tick=0,
            new_scientific_candidates=0, admissions=1, launches=1, archives_this_tick=0,
            lifecycle_guards=29, additive_guards=8, admission_guards=14,
            provenance_correction=y96["provenance_correction"], source_selection=selection,
            canonical_dispatch=dispatch, original_Y94_archive_preserved=y94["terminal_evidence"],
            formal_FIRST_or_cost_executed=False, efficacy_unknown=True,
            evidence=[ref(__file__), selection, dispatch, ref(y96["admission_receipt"]),
                y96["post_training_freeze"], y95["terminal_handoff_plan"], ref(guardlog)]
                + [row["evidence"] for row in observations.values()] + list(occupations.values())))
        memory = ROOT / "pipeline/codex_memory.md"
        backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
        assert not backup.exists()
        original = memory.read_bytes()
        backup.write_bytes(original)
        assert backup.read_bytes() == original
        lines = [
            "# AdaRES_project Research Memory",
            "## Goal And Phase",
            "- Current token " + token + "; snapshot " + view["generated_at"] + "; deadline " + view["deadline_at"] + "; never extend.",
            "- Actual CODEX_THREAD_ID=owner01a07e32-78bf-7670-925b-370911ea48db; background Pipeline-AdaRES_project Disabled; experiment_first enabled.",
            "- Only server1GPU0 existing350W and server2GPU0 fixed280W, one independent experiment/card. ALL server2GPU1 use paused; Y57 separately paused at original epoch27. No server6/cross-project/scheduler/power/foreign-job changes.",
            "- Goal SHA" + GOAL + "; GPU1 pause authority SHA" + PAUSE + "; original goal bytes and every old signed receipt preserved.",
            "- PROJECT_ROOT=D:/project; PYTHONDONTWRITEBYTECODE=1; PYTHONUTF8=1; python -B; SSH keys work, no private credential read.",
            "- Exact previous seven-section memory: " + evidence(ref(backup)) + "; complete historical snapshot chain byte-preserves ALL mechanisms, bans, failures, controls, protocols, authorizations, receipts and dedicated restore requirements; compression never grants reopening.",
            "## Champion And Baseline",
            "- Target .5150; formal FIRST Y36=.510725438294, W2=.510107995189, B1=.502221764356, Y52=.509708428366, Y54=.506342952986. Historical CSV Y36 .5110 and training best are not replacement FIRST.",
            "- Full original review:9508 expressions/141 scenes, candidate+Y36/W2/B1 twice each, FIRST primary, scene bootstrap10000/seed1999, TRAIN cuts1487/3325; all subgroup harm/mIoU/Acc25/50/activequeries/latency/memory/original-device cost unchanged.",
            "- Y54 large-object harmCI[-.02678985,-.00070265], Y52 overallCI crosses0; one seed cannot estimate seed variance. No formal FIRST/cost/promotion this tick.",
            "## Active Hypotheses",
            "- Y96 independent W2 original sampled-seed initial-query bypass q_init=original five-layer generator(seed)+seed@zeroW256x256 once; +65536/total164695173/no RNG. Same undetached post-SCG sampled128x256 seed, complete original data/full inputs/2D/backbone/sampling/SCG/hard gates/static masks/six refinements/shared heads/targets-loss-selector/seed1999/fixed30/native YAML unchanged.",
            "- Y96 not Y73 repeated anchoring/Y75 language/Y77 coordinates/Y84 scene mean/Y95 recurrent query mean/Y93 confidence bypass; no combination, defect/novelty/benefit/reopening claim. Original CPU controls and all mechanism details preserved in exact backup; never replay.",
            "- Y96 actual original native server1GPU0/350W PASS was PREVIOUS tick, reused without replay: " + evidence(y96["native_smoke_result"]) + "; losses[4.615510940551758,6.309134006500244], peak_reserved6759120896; two finite real TRAIN backward/optimizer steps, original generator/initial-coarse/shared heads/hard gates verified, no efficacy claim.",
            "- Production HEADY96=" + Y96 + "; ONLY seven already native-tested additions selected " + evidence(selection) + "; unrelated index/working files preserved; only server1 safely synchronized.",
            "- Y96 actual14 admission guards passed this tick; COMPLETE ORIGINAL Y94 server1GPU0/350W training/evaluation/cost freeze " + evidence(y96["post_training_freeze"]) + "; no server2 substitution.",
            "- Current Y96 receipt " + evidence(ref(y96["admission_receipt"])) + "; provenance correction " + evidence(y96["provenance_correction"]) + " fixes the cross-tick command token ONLY. Original receipt/smoke byte-preserved; no scientific value/code/runtime/PID change or native replay.",
            "- Canonical Y96 dispatch " + evidence(dispatch) + "; launch proof and fresh native counter increase verified. Future mutation must preserve current receipt, source and entire original protocol.",
            "- Y98 CPU-prepared commit03799b0b2cacca31987a604c539a4457498175a1/full existing1024 appearance-to-static-mask bypass; Y99 commite5151a722778af3eecb28d0d4f6ae75e18a60e0c/full32 sparse-input mask-MLP bypass. CPU evidence/source/native bodies frozen in backup; no GPU/admission. Y98 thenY99 need NEW tested same-byte HEADY96/current-owner bindings before use; old runners/parents are not authority.",
            "## Closed Or Failed Directions",
            "- Full historical rejected/closed mechanisms and original evidence are in exact backup chain. Preserve Y53/Y55/Y50/Y51/Y49rev1/Y46/Y48/F1 rejection and goal bans; no validation scans of thresholds/temperatures/margins/occupancy/loss/features/epochs.",
            "- Y97 original and SAME-ID revision1 CPU numerical failures remain immutable implementation holds. No replay, renamed retry, tolerance/equality relaxation or hypothesis rejection. Y86 original GPU native numerical failure also unchanged. No timer or generic resource grant authorizes reopening.",
            "- Preserve prior narrow observer/runtime/metadata repairs and all negative logs, Y22/Y23 dedicated restoration, Y57 epoch27 model/optimizer543/scheduler27 pause. Archived Y81/Y82/Y83/Y91/Y92/Y93/Y94 remain products, never restart or replay terminal helpers.",
            "## Running And Queue",
            "- Current production observer=y96_deployment.py observe --id AdaRES-Y95|AdaRES-Y96 --token CURRENT --label NEW; older deployment observers reject HEADY96. Do not live-observe archived Y94.",
        ]
        for name, cid, pid, commit in RUNS:
            row, phase = observations[cid], observations[cid]["phase"]
            lines.append("- " + cid + " original PID" + str(pid) + "/" + name + "GPU0, "
                + phase["phase"] + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/"
                + str(phase["count"]) + "; native counters/source/YAML/runtime/power verified "
                + evidence(row["evidence"]) + "; commit" + commit + ".")
        lines += [
            "- Y95 ORIGINAL exact future terminal directory20261109_081912, NOT launch20261109_212023. NEW actual-nativeY96 compatibility " + evidence(y95["terminal_handoff_plan"]) + ";29 lifecycle+8 additive guards. Future helper finalize_y95_fixed30_y96.py at HEADY95/Y96 only after natural completion; no Y95 checkpoint loaded/archived.",
            "- Y94 exact original directory20261109_111838 completed PREVIOUS tick, full Train/Val1..30 stable finite last30/best21/model/optimizer544/scheduler30-21 verified ONCE and canonical archive " + evidence(y94["terminal_evidence"]) + ";trainingbest.5075652599334717 is NOT formal FIRST. Full original server1/350W review waits while Y96 owns GPU0.",
            "- Y93 original20261108_082521 verified/archived ONCE, last30/best20/model/optimizer544/scheduler30-20, trainingbest.5050445795059204 not FIRST; original server2/280W full review waits onY95. Y92/Y91 and earlier archives/protocols unchanged in backup.",
            "- Fresh canonical physical occupations " + "; ".join(evidence(value) for value in occupations.values()) + "; both authorizedGPU0 busy+reserved; no free exclusive resource. All original GPU1 frozen FIRST/cost stay on user hold, never migrate/substitute.",
            "- Server1=songcl@192.168.1.175/dl02-MS-7B73, /data1/songcl/Project/AdaRES_project/AdaRES, private runtime/adares_project/bin/python and cuda113lib64, exact ../data/, only two existing SERVERS mappings. Server2=songcl@192.168.1.163/dl01-MS-7C60, /bak_home/songcl/Project/AdaRES_project/AdaRES, /home/songcl/.conda/envs/adares_project/bin/python, exact /bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/.",
            "## Open Questions And Risks",
            "- Y96 native feasibility and increasing training counters do not prove efficacy/full-data memory; all original FIRST/cost/subgroup uncertainty obligations remain. Y57 and all GPU1 use require separate explicit reauthorization; no new user action requested.",
            "- Next tick first observe original Y95/Y96 and dispatch qualified waiting work. Y94 terminal/nativeY96 must NOT replay. Y95 natural completion requires exact finalize_y95_fixed30_y96.py verification/archive; then fresh exclusive authorized resources and NEW tested bindings before the remaining Y98/Y99 native tasks. Y96 future exact terminal plan still needs preparation at current HEAD, without loading live checkpoints.",
            "## Last Tick",
            "- " + token + ": original Y95 progress verified;29 lifecycle+8 actual-native additive guards passed, Y96 exact seven-file source selection,14 admission guards/full original server1 freeze/receipt, canonical launch and actual native counter increase verified. Prior Y94 archive/Y96 native reused without replay.",
            "- Native command token provenance corrected additively, preserving original receipt/smoke, all scientific fields and PID. ScientificIDs0/scientificCPUattempts0/nativeGPUtasks0/admissions1/launches1/archives0/FIRST0/cost0. Full original data/runtime/power/protocol/bans/foreign jobs and both pauses preserved.",
            "- Closeout " + evidence(ref(output)) + "; official report/finish result and released receipt are authoritative; never claim finish before it actually passes.",
        ]
        assert len(lines) < 120 and sum(line.startswith("## ") for line in lines) == 7
        memory.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        phases = []
        for cid, row in observations.items():
            phase = row["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30 step"
                          + str(phase["step"]) + "/" + str(phase["count"]))
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅AdaRES_project，核验实际owner、Disabled后台调度和最新资源暂停后begin；先观察Y95并派发已有合格任务，无ready waiting。承接已完成Y94精确归档和Y96实际native结果，均未重放。",
            "完成：Y95新版兼容绑定29项原生命周期及8项actual-native源码检查通过；Y96仅七个已测试新增文件进入HEAD，其他index/工作文件不动，仅向实际目标server1安全同步。",
            "新准备：14项准入检查通过；完整继承Y94原server1GPU0/350W训练/FIRST/成本/有害子组协议，生成收据并当轮canonical派发Y96，已用递增原生训练步数确认运行。",
            "运行：" + "；".join(phases) + "。原PID/source/YAML/签名收据/私有runtime和350W/280W均核验，两张授权GPU0物理busy+reserved。",
            "发现跨tick复用native的命令token被准入工具错标为当前tick，另存来源更正smoke和收据并验证；旧smoke/旧收据和所有实测字段字节保持，未重跑native、修改模型或干预训练。",
            "本轮新科学ID/科学CPU尝试/nativeGPU任务/归档/FIRST/成本均0，准入1、启动1；精确备份七节记忆、保存真实证据并按官方report/finish收尾，以实际收据和释放结果为准。",
        ], decisions=[
            "已有Y98/Y99 CPU准备继续保留，不能用旧HEAD/旧owner直接执行，后续须NEW tested同字节绑定；Y97负证据和数值门槛保持，不凑额度新增或重测。",
            "全部GPU1及原GPU1冻结后评/成本继续暂停，Y57独立暂停；禁止server6、跨项目、迁移替代、调度/功耗变更和干预外部任务。原goal、旧签名收据、原数据/私有runtime/部署映射、seed1999固定30轮和完整FIRST/成本协议不变。",
        ], evidence=[ref(output), ref(backup), y96["provenance_correction"]]))
        assert all(ref(path) == entry for path, entry in before.items())
        assert git("ls-files", "--stage", "-z") == index
        return output, notes, len(lines)
    finally:
        writer.close()


def finish(token):
    lease(token, 300)
    for _, cid, _, _ in RUNS:
        command(["scripts/y96_deployment.py", "observe", "--id", cid,
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
    print(json.dumps(dict(closeout=ref(output), status=receipt["status"], postflight=receipt["postflight"],
        context_committed=True, lease_release=receipt["lease_release"], memory_lines=lines,
        new_candidates=0, native_GPU_tasks=0, admissions=1, launches=1, archives=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    finish(parser.parse_args().token)
