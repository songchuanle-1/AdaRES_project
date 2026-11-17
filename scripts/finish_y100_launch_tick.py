"""Close the verified Y98 archive and actual-native Y100 continuation."""
import argparse
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y100_deployment import Y98, Y99, Y100, git, require_running_parent_handoff
from advance_y100 import prepared, validate_result, make_freeze
import closure
import gpu_occupation

RUNS = (("server-1", "AdaRES-Y99", Y99), ("server-2", "AdaRES-Y100", Y100))


def evidence(r):
    return r["path"] + " SHA" + r["sha256"]


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y100
        before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        assert before["pipeline/goal.json"]["sha256"] == GOAL
        assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == PAUSE
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {r["id"]: r for r in work["candidates"]}
        y98, y99, y100 = (candidates["AdaRES-" + n] for n in ("Y98", "Y99", "Y100"))
        queue = monitor.load_queue(str(ROOT))
        assert monitor.load_running(str(ROOT)) == {"server-1": {"0": "AdaRES-Y99"}, "server-2": {"0": "AdaRES-Y100"}}
        context = tick.load(ROOT / "pipeline/headless/.tick_context.json")
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        assert not view["preflight"]["operational_issues"]
        terminal = tick.load(closure.artifact(ROOT, y98["terminal_evidence"]))
        assert terminal["token"] == token and terminal["id"] == "AdaRES-Y98"
        assert terminal["process_absent"] is True and terminal["CUDA_initialized"] is False
        assert terminal["completed_epochs"] == {p: list(range(1, 31)) for p in ("Train", "Val")}
        assert [r["epoch"] for r in terminal["checkpoints"]] == [30, 20]
        assert all(r["model_finite"] and r["optimizer_finite"] and r["optimizer_state_count"] == 544 and r["scheduler_last_epoch"] == r["epoch"] for r in terminal["checkpoints"])
        assert all("/20261110_080627/" in r["path"] for r in terminal["checkpoints"])
        assert y98["stage"] == "blocked" and y98["blocker_kind"] == "exclusive_gpu_slot" and y98["scientific_status"] == "fixed30_complete_frozen_review_pending"
        candidate, _, _, frozen = prepared(work)
        assert candidate["id"] == y100["id"]
        raw = tick.load(closure.artifact(ROOT, y100["native_smoke_result"]))
        validate_result(raw, y100, frozen)
        assert raw["losses"] == [4.615511417388916, 6.338587284088135] and raw["peak_reserved_bytes"] == 6819938304
        receipt_ref = ref(y100["admission_receipt"])
        receipt = tick.load(closure.artifact(ROOT, receipt_ref))
        smoke = tick.load(closure.artifact(ROOT, receipt["native_smoke"]))
        assert smoke["command"] == "python -B scripts/run_y100_native_v2.py --server server-2 --token " + token
        freeze = tick.load(closure.artifact(ROOT, y100["post_training_freeze"]))
        parentref = y98["post_training_freeze"]
        expected = make_freeze(tick.load(closure.artifact(ROOT, parentref)), parentref)
        for k in ("training", "evaluation", "resource_condition", "resource_authorization", "parent_protocol"):
            assert freeze[k] == expected[k]
        require_running_parent_handoff(work, y100)
        handoff = tick.load(closure.artifact(ROOT, y99["terminal_handoff_plan"]))
        assert handoff["token"] == token and handoff["tested_additive_native_evidence"] == y100["native_smoke_result"]
        assert handoff["payload"]["allowed_markers"] == [Y99, Y100]
        assert handoff["CPU_guard_tests_passed"] == 30 and handoff["additive_guard_tests_passed"] == 8
        assert handoff["full_terminal_verification_executed"] is False and handoff["checkpoint_loaded"] is False
        guards = ref("pipeline/checks/adares-y100_admission_guards_" + token + ".log")
        assert "Ran 11 tests" in closure.artifact(ROOT, guards).read_text(encoding="utf-8")
        dispatch_ref = ref("pipeline/checks/adares-y100_dispatch_" + token + ".json")
        dispatch = tick.load(closure.artifact(ROOT, dispatch_ref))
        assert dispatch["task"]["status"] == "running" and dispatch["task"]["launch_verified"] is True
        observations, occupations = {}, {}
        for name, cid, commit in RUNS:
            task = next(r for r in queue["gpu_queues"][name] if r["id"] == cid)
            assert (task["status"], task["launch_verified"], task["code_commit"], str(task["gpu"])) == ("running", True, commit, "0")
            assert task["pid"] == (2980836 if cid == "AdaRES-Y99" else dispatch["task"]["pid"])
            closure.admission(ROOT, task)
            entry = candidates[cid]["latest_training_observation"]
            value = tick.load(closure.artifact(ROOT, entry))
            assert value["token"] == token and value["native_progress_confirmed"] is True and value["pid"] == task["pid"]
            assert value["source_marker"] in (commit, Y100)
            fresh(value, "observed_local")
            observations[cid] = dict(evidence=entry, phase=value["after"]["latest"], pid=task["pid"])
            proof = ref("pipeline/checks/gpu_occupation_" + name + "_release_" + token + ".json")
            fresh(tick.load(closure.artifact(ROOT, proof)), "observed_at")
            held = candidates["AdaRES-Y96"] if name == "server-1" else y98
            server_view = next(r for r in context["queue"]["servers"] if r["name"] == name)
            state = gpu_occupation.validated_state(ROOT, context, held, server_view, queue)
            assert state["busy_gpu_ids"] == state["reserved_gpu_ids"] == [0] and not state["idle_gpu_ids"]
            occupations[name] = proof
        pause = next(r for r in queue["gpu_queues"]["server-2"] if r["id"] == "AdaRES-Y57")
        assert pause["status"] == "paused" and pause["dispatch_hold"] and pause["resume_requires_user_authorization"]
        selection = ref("pipeline/checks/adares-y100_source_selection_" + token + ".json")
        helpers = ref("pipeline/checks/y100_launch_helpers_" + token + ".json")
        old_observer_failure = ref("pipeline/checks/production_adares-y98_" + token + "_start.log")
        out = ROOT / ("pipeline/checks/y100_launch_closeout_" + token + ".json")
        assert not out.exists()
        tick.save(out, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            observations=observations, occupations=occupations, original_Y98_terminal=y98["terminal_evidence"],
            original_Y99_handoff=y99["terminal_handoff_plan"], actual_Y100_native=y100["native_smoke_result"],
            source_selection=selection, dispatch=dispatch_ref, source_helpers=helpers,
            preserved_Y98_live_observer_failure=old_observer_failure,
            observer_failure_resolution="Exact original PID absent; full fixed30 original checkpoints subsequently verified and archived. No restart or weakening of observation.",
            new_scientific_candidates=0, scientific_CPU_replays=0, native_GPU_tasks=1, admissions=1, launches=1, archives=1,
            original_terminal_lifecycle_tests=29, additive_terminal_lifecycle_tests=30, additive_terminal_tests=8, admission_guard_tests=11,
            full_original_protocol_preserved=True, formal_FIRST_or_cost_executed=False,
            evidence=[ref(__file__), selection, dispatch_ref, receipt_ref, y100["post_training_freeze"], y98["terminal_evidence"],
                y99["terminal_handoff_plan"], guards, helpers, old_observer_failure] + [r["evidence"] for r in observations.values()] + list(occupations.values())))
        memory = ROOT / "pipeline/codex_memory.md"
        backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
        assert not backup.exists()
        old_bytes = memory.read_bytes()
        backup.write_bytes(old_bytes)
        assert backup.read_bytes() == old_bytes
        old = old_bytes.decode("utf-8").splitlines()
        sections = {}
        current = None
        for line in old:
            if line.startswith("## "):
                current = line
                sections[current] = []
            elif current:
                sections[current].append(line)
        assert len(sections) == 7
        phases = {cid: r["phase"]["phase"] + str(r["phase"]["epoch"]) + "/30 step" + str(r["phase"]["step"]) + "/" + str(r["phase"]["count"]) for cid, r in observations.items()}
        lines = ["# AdaRES_project Research Memory", "## Goal And Phase",
            "- Current token " + token + "; snapshot " + view["generated_at"] + "; deadline " + view["deadline_at"] + "; never extend.",
            "- Actual CODEX_THREAD_ID=owner01a07e32-78bf-7670-925b-370911ea48db; Pipeline-AdaRES_project Disabled; experiment_first enabled. PROJECT_ROOT=D:/project; PYTHONDONTWRITEBYTECODE=1; PYTHONUTF8=1; python -B; SSH keys only, no credential read.",
            "- Only server1GPU0 existing350W and server2GPU0 fixed280W, one experiment/card; ALL server2GPU1 use paused, Y57 separately paused epoch27. No server6/cross-project/scheduler/power/foreign-job changes.",
            "- Goal SHA" + GOAL + "; pause SHA" + PAUSE + "; original scientific goal/receipts/protocols/data/mappings/bans byte-preserved.",
            "- Exact previous seven-section memory: " + evidence(ref(backup)) + "; complete historical sources, evidence, negative records and dedicated recoveries preserved through exact backup chain; never reopening authority.",
            "## Champion And Baseline"] + sections["## Champion And Baseline"] + ["## Active Hypotheses",
            "- Production HEADY100=" + Y100 + "; exact7 actual-native-tested Y100 additions selected " + evidence(selection) + "; parentY99=" + Y99 + "; production source selection preserves unrelated index/working files and all original Y99 files; only server2 synchronized.",
            "- Y99 independent W2 full32 sparse-input linear bypass beside original x_mask MLP; +8192/noRNG. Original projections/full inputs/static7/shared confidence/hard gates/query generator/refinements/data/seed1999/fixed30 unchanged. Original first CPU/native/receipt/full server1 protocol preserved in exact backup, no replay.",
            "- Y100 independent W2 full1024 raw-appearance linear bypass beside complete original input_proj_2d BEFORE original sum_norm/visual memory; +262144/total164891781/noRNG. All original projection/static mask7/full32spatial/full1024appearance/shared confidence/refinement/SCG/hard gates/loss/selector/data/seed1999fixed30 preserved. Not Y98 appearance-to-mask/Y99 spatial-to-mask/Y71 product/Y97 language-square; no tuning/combination/reopening.",
            "- Y100 FIRST original CPU result " + evidence(y100["CPU_result"]) + "; CPU version " + evidence(y100["source_manifest"]) + ";10 evidence+8 corrected runtime guards/native plans " + evidence(y100["native_preparation"]) + "; exact original CPU/scientific files reused without replay. Preserved first metadata KeyError earlier_Y94_preserved and v2 original-owner correction remain in backup and original preparation; never use first runner/plans.",
            "- Y100 FIRST actual server2GPU0/280W native PASS " + evidence(y100["native_smoke_result"]) + "; losses[4.615511417388916,6.338587284088135], peak_reserved6819938304; original full1024 appearance/projection/sum_norm/first visual memory/static7/shared head/hard gates/14 original visual parameter gradients and two finite real TRAIN backward-updates verified. No efficacy/full-data memory proof.",
            "- Y10011 actual-native admission guards PASS; full COMPLETE ORIGINAL Y98 server2GPU0/280W training/FIRST/subgroups/cost protocol freeze " + evidence(y100["post_training_freeze"]) + "; receipt " + evidence(receipt_ref) + "; source helpers " + evidence(helpers) + "; actual launch " + evidence(dispatch_ref) + ". No formal FIRST/cost/promotion.",
            "## Closed Or Failed Directions"] + sections["## Closed Or Failed Directions"] + ["## Running And Queue",
            "- Current observer=y100_deployment.py observe --id AdaRES-Y99|AdaRES-Y100 --token CURRENT --label NEW; earlier observers reject HEADY100. Do not live-observe/rearchive Y98/Y96/Y95 or earlier archives.",
            "- AdaRES-Y99 original PID2980836/server1GPU0/350W, " + phases["AdaRES-Y99"] + "; original source/YAML/private runtime/power/increasing counters " + evidence(observations["AdaRES-Y99"]["evidence"]) + "; original receipt/freeze unchanged.",
            "- AdaRES-Y100 original PID" + str(observations["AdaRES-Y100"]["pid"]) + "/server2GPU0/280W, " + phases["AdaRES-Y100"] + "; launch " + evidence(dispatch_ref) + "; native counter increase " + evidence(observations["AdaRES-Y100"]["evidence"]) + "; exact original checkpoint directory must be discovered read-only before terminal preparation.",
            "- Y98 exact original20261110_080627 naturally completed; absentPID25962/full Train-Val1..30/stable finite last30-best20/model/optimizer544/schedulers30-20 verified ONCE/canonical archive " + evidence(y98["terminal_evidence"]) + "; trainingbest.49940434098243713 is NOT formal FIRST. Complete original server2/280W review now waits onY100; original artifacts/receipt/freeze retained, no restart/migration/substitution/terminal replay.",
            "- Y99 exact original20261111_180638 (not launch180628); NEW actual-Y100-native compatible30+8 terminal plan " + evidence(y99["terminal_handoff_plan"]) + "; future finalize_y99_fixed30_y100.py ONLY at HEADY99/Y100 after natural30/exact absent original PID/full Train-Val1..30/stable finite last-best-model-optimizer-scheduler and original source/private runtime/350W. No live Y99 checkpoint loaded.",
            "- Y96 original20261110_165702 complete archived ONCE, last30/best23/optimizer544/scheduler30-23, trainingbest.5081601738929749 NOT FIRST; full original server1/350W review waits onY99. Y95 original20261109_081912 last30/best20/trainingbest.508540153503418; Y94/Y93/Y92/Y91/Y81/Y82/Y83 earlier verified products and all exact proofs remain in backup chain. Never restart/replay terminal helpers.",
            "- Fresh canonical occupation " + "; ".join(evidence(r) for r in occupations.values()) + "; both authorizedGPU0 busy/reserved by originalY99/newY100. Original GPU1 full frozen FIRST/cost remain user-held, never moved or substituted.",
            "- Server1=songcl@192.168.1.175/dl02-MS-7B73, /data1/songcl/Project/AdaRES_project/AdaRES, private runtime/adares_project/bin/python+cuda113lib64, exact ../data/, only original two SERVERS mappings. Server2=songcl@192.168.1.163/dl01-MS-7C60, /bak_home/songcl/Project/AdaRES_project/AdaRES, /home/songcl/.conda/envs/adares_project/bin/python, exact /bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/.",
            "## Open Questions And Risks",
            "- No formal FIRST/cost/promotion. Trainingbest/native feasibility/increasing counters do not prove efficacy. GPU1 all-use/Y57 separate pause remain; no new user action needed.",
            "- Y98 start live-observer assertion failure retained " + evidence(old_observer_failure) + "; subsequently confirmed exact original process absent and FULL original checkpoint terminal proof/canonical archive completed; no restart or relaxed observation.",
            "- Next tick first observe Y99/Y100, dispatch qualified waiting work, implement independent variants per snapshot if short. Prepare exact original Y100 terminal tools with read-only actual save-directory discovery; never infer from launch name or load live weights. Y99 uses new tested finalize_y99_fixed30_y100.py only after original natural30/full terminal proof. No Y100 native/old CPU/version/plan/binding or archived terminal replay.",
            "## Last Tick",
            "- " + token + ": Y98 natural fixed30 full finite exact original checkpoint proof/canonical archive; Y100 FIRST native PASS, original Y99 actual-native compatible30+8, exact source selection,11 actual-evidence admission guards/full original server2 freeze/receipt/canonical dispatch/increasing counters verified. Original Y99 continues unchanged.",
            "- New scientificIDs0/CPUscientificreplays0/nativeGPUtasks1/admissions1/launches1/archives1/FIRST0/cost0. Goal/old receipts/data/runtime/mappings/seed1999fixed30/350W280W/bans/GPU1+Y57 pauses/foreign jobs intact. Closeout " + evidence(ref(out)) + "; official finish/receipt/release authoritative."]
        assert len(lines) < 120 and sum(r.startswith("## ") for r in lines) == 7
        memory.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "运行：" + "；".join(cid + " " + p for cid, p in phases.items()) + "。原Y99继续，Y100在server2GPU0/280W按原生YAML固定30轮运行，精确源码/数据/私有环境/功耗和原生步数递增已验证。",
            "完成：Y98自然固定30轮结束，原目录20261110_080627完整Train/Val1..30及稳定有限last30/best20/model/optimizer544/scheduler30-20已核验并规范归档，原产物保留；训练best.499404不是正式FIRST。首次活跃进程检查失败原日志保留，随后精确终态检查确认自然结束，不重启、不从日志推断完整性。",
            "新准备并推进：Y100首次原生TRAIN两步检查通过，loss[4.615511417388916,6.338587284088135]，峰值reserved6819938304；完整原外观投影/sum_norm/视觉记忆/原静态七次readout/共享head/原hard-gates及新旧梯度通过。原CPU与科学版本不重跑。原Y99终态兼容30+8检查、11项实际证据准入检查、完整原server2后评协议冻结/收据/规范派发完成。",
            "本轮新科学ID0、CPU科学重跑0、原生GPU检查1、准入1、新训练1、归档1。无正式FIRST、成本评测或晋升。"
        ], decisions=[
            "保留原Y99和新Y100固定30轮训练及所有原产物/收据/精确数据/私有环境/路径映射/seed1999/科学禁用项。Y98完整原设备冻结后评等待server2空闲。下一轮按实际保存声明准备Y100精确终态工具，不推断目录或加载活跃权重。",
            "server2GPU1全部使用及Y57单独暂停保持；不迁移冻结后评/成本，不重启受保护实验，不使用server6、不跨项目、不改调度/功耗、不干预外部任务。无新增用户处理事项。"
        ], evidence=[ref(out), ref(backup)]))
        assert all(ref(p) == r for p, r in before.items()) and git("ls-files", "--stage", "-z") == index
        return out, notes, len(lines)
    finally:
        writer.close()


def finish(token):
    lease(token, 240)
    for _, cid, _ in RUNS:
        command(["scripts/y100_deployment.py", "observe", "--id", cid, "--token", token, "--label", "release"])
    for name, _, _ in RUNS:
        command([SKILL + "/gpu_occupation.py", "AdaRES_project", "--token", token, "--server", name, "--label", "release"])
    out, notes, lines = record(token)
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
    print(json.dumps(dict(closeout=ref(out), status=receipt["status"], postflight=receipt["postflight"], context_committed=True,
        lease_release=receipt["lease_release"], memory_lines=lines, observations={cid: r["phase"] for cid, r in tick.load(out)["observations"].items()},
        native_GPU_tasks=1, admissions=1, launches=1, archives=1)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True)
    finish(p.parse_args().token)
