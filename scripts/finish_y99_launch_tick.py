"""Close exact Y96 archival and actual-native Y99 admission/launch."""
import argparse
import copy
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y99_deployment_v2 import Y96, Y98, Y99, git, require_running_parent_handoff
from advance_y99_v2 import prepared, validate_result, make_freeze
import closure
import gpu_occupation

RUNS = (("server-1", "AdaRES-Y99", Y99), ("server-2", "AdaRES-Y98", Y98))


def evidence(row):
    return row["path"] + " SHA" + row["sha256"]


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y99
        before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        assert before["pipeline/goal.json"]["sha256"] == GOAL
        assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == PAUSE
        queue = monitor.load_queue(str(ROOT))
        assert monitor.load_running(str(ROOT)) == {"server-1": {"0": "AdaRES-Y99"}, "server-2": {"0": "AdaRES-Y98"}}
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {r["id"]: r for r in work["candidates"]}
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        context = tick.load(ROOT / "pipeline/headless/.tick_context.json")
        assert not view["preflight"]["operational_issues"]
        y96, y98, y99 = (candidates["AdaRES-" + n] for n in ("Y96", "Y98", "Y99"))
        terminal = tick.load(closure.artifact(ROOT, y96["terminal_evidence"]))
        assert terminal["token"] == token and terminal["id"] == "AdaRES-Y96"
        assert terminal["completed_epochs"] == {p: list(range(1, 31)) for p in ("Train", "Val")}
        assert terminal["process_absent"] is True and terminal["CUDA_initialized"] is False
        assert [r["epoch"] for r in terminal["checkpoints"]] == [30, 23]
        assert all(r["model_finite"] and r["optimizer_finite"] and r["optimizer_state_count"] == 544
                   and r["scheduler_last_epoch"] == r["epoch"] for r in terminal["checkpoints"])
        assert y96["scientific_status"] == "fixed30_complete_frozen_review_pending" and y96["stage"] == "blocked"
        assert y96["blocker_kind"] == "exclusive_gpu_slot"
        c, _, _, frozen = prepared(work)
        assert c["id"] == y99["id"]
        raw = tick.load(closure.artifact(ROOT, y99["native_smoke_result"]))
        validate_result(raw, y99, frozen)
        assert raw["losses"] == [4.615511417388916, 6.3237128257751465]
        assert raw["peak_reserved_bytes"] == 6769606656
        receipt_ref = ref(y99["admission_receipt"])
        receipt = tick.load(closure.artifact(ROOT, receipt_ref))
        smoke = tick.load(closure.artifact(ROOT, receipt["native_smoke"]))
        assert smoke["command"] == "python -B scripts/run_y99_bound_native_y98.py --server server-1 --token " + token
        freeze = tick.load(closure.artifact(ROOT, y99["post_training_freeze"]))
        parentref = y96["post_training_freeze"]
        parent = tick.load(closure.artifact(ROOT, parentref))
        expected = make_freeze(parent, parentref)
        for k in ("training", "evaluation", "resource_condition", "resource_authorization", "parent_protocol"):
            assert freeze[k] == expected[k]
        require_running_parent_handoff(work, y99)
        handoff = tick.load(closure.artifact(ROOT, y98["terminal_handoff_plan"]))
        assert handoff["token"] == token and handoff["tested_additive_native_evidence"] == y99["native_smoke_result"]
        assert handoff["payload"]["allowed_markers"] == [Y98, Y99]
        assert handoff["CPU_guard_tests_passed"] == 29 and handoff["additive_guard_tests_passed"] == 8
        guardlog = ref("pipeline/checks/adares-y99_admission_guards_" + token + ".log")
        assert "Ran 11 tests" in closure.artifact(ROOT, guardlog).read_text(encoding="utf-8")
        dispatch_ref = ref("pipeline/checks/adares-y99_dispatch_" + token + ".json")
        dispatch = tick.load(closure.artifact(ROOT, dispatch_ref))
        assert dispatch["task"]["status"] == "running" and dispatch["task"]["launch_verified"] is True
        observations, occupations = {}, {}
        for name, cid, commit in RUNS:
            task = next(r for r in queue["gpu_queues"][name] if r["id"] == cid)
            assert (task["status"], task["launch_verified"], task["code_commit"], str(task["gpu"])) == ("running", True, commit, "0")
            assert task["pid"] == (dispatch["task"]["pid"] if cid == "AdaRES-Y99" else 25962)
            closure.admission(ROOT, task)
            entry = candidates[cid]["latest_training_observation"]
            value = tick.load(closure.artifact(ROOT, entry))
            assert value["token"] == token and value["native_progress_confirmed"] is True
            assert value["pid"] == task["pid"] and value["source_marker"] in (commit, Y99)
            fresh(value, "observed_local")
            observations[cid] = dict(evidence=entry, phase=value["after"]["latest"], pid=task["pid"])
            proof_ref = ref("pipeline/checks/gpu_occupation_" + name + "_release_" + token + ".json")
            fresh(tick.load(closure.artifact(ROOT, proof_ref)), "observed_at")
            held = next(r for r in work["candidates"] if r.get("stage") == "blocked"
                        and r.get("blocker_kind") == "exclusive_gpu_slot" and r.get("required_server") == name)
            server = next(r for r in context["queue"]["servers"] if r["name"] == name)
            state = gpu_occupation.validated_state(ROOT, context, held, server, queue)
            assert state["busy_gpu_ids"] == state["reserved_gpu_ids"] == [0] and not state["idle_gpu_ids"]
            occupations[name] = proof_ref
        paused = next(r for r in queue["gpu_queues"]["server-2"] if r["id"] == "AdaRES-Y57")
        assert paused["status"] == "paused" and paused["dispatch_hold"] and paused["resume_requires_user_authorization"]
        selection = ref("pipeline/checks/adares-y99_source_selection_" + token + ".json")
        drafts = ref("pipeline/checks/y99_launch_helpers_" + token + ".json")
        corrected = ref("pipeline/checks/y99_launch_helpers_" + token + "_v2.json")
        output = ROOT / ("pipeline/checks/y99_launch_closeout_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            observations=observations, occupations=occupations, original_Y96_terminal=y96["terminal_evidence"],
            original_Y98_handoff=y98["terminal_handoff_plan"], actual_Y99_native=y99["native_smoke_result"],
            source_selection=selection, dispatch=dispatch_ref, preserved_untested_drafts=drafts,
            corrected_operations=corrected, new_scientific_candidates=0, scientific_CPU_replays=0,
            native_GPU_tasks=1, admissions=1, launches=1, archives=1,
            terminal_lifecycle_tests=29, additive_terminal_tests=8, admission_guard_tests=11,
            formal_FIRST_or_cost_executed=False, full_original_protocol_preserved=True,
            wording_correction="Old terminal binding's only_exact_Y94_archived and original native runner's appearance wording are stale template labels; actual IDs, original Y96 checkpoint proof and validated Y99 full32 spatial paths are authoritative.",
            evidence=[ref(__file__), selection, dispatch_ref, receipt_ref, y99["post_training_freeze"],
                      y96["terminal_evidence"], y98["terminal_handoff_plan"], guardlog, drafts, corrected]
                + [r["evidence"] for r in observations.values()] + list(occupations.values())))
        memory = ROOT / "pipeline/codex_memory.md"
        backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
        assert not backup.exists()
        old_bytes = memory.read_bytes()
        backup.write_bytes(old_bytes)
        assert backup.read_bytes() == old_bytes
        old = old_bytes.decode("utf-8").splitlines()
        lines, updated = [], set()
        for line in old[:old.index("## Last Tick")]:
            if line.startswith("- Current token "):
                line = "- Current token " + token + "; snapshot " + view["generated_at"] + "; deadline " + view["deadline_at"] + "; never extend."
                updated.add("token")
            elif line.startswith("- Exact previous seven-section memory:"):
                line = "- Exact previous seven-section memory: " + evidence(ref(backup)) + "; all historical routes/protocols/negative evidence/bans/dedicated recovery byte-preserved; never reopening authority."
                updated.add("backup")
            elif line.startswith("- Production HEADY98="):
                line = "- Production HEADY99=" + Y99 + "; exact seven original CPU/native-tested Y99 additions selected " + evidence(selection) + "; all unrelated index/working files and original Y98 preserved; only server1 synchronized. Prior Y98 source selection and all originals preserved in exact backup chain."
                updated.add("head")
            elif line.startswith("- Y99 full32 "):
                line += " FIRST actual server1GPU0/350W native PASS " + evidence(y99["native_smoke_result"]) + "; losses[4.615511417388916,6.3237128257751465], peak_reserved6769606656, original full32/base/spatial-appearance projection/seven static consumers/shared heads/hard gates and two real TRAIN backward-updates verified.11 admission guards/full original Y96 server1 protocol freeze " + evidence(y99["post_training_freeze"]) + "; receipt " + evidence(receipt_ref) + "; original CPU/binding evidence reused without replay."
                line = line.replace("No CPU scientific replay/native GPU/admission/launch; future runner=run_y99_bound_native_y98.py.", "Original preparatory binding remains immutable; runner=run_y99_bound_native_y98.py.")
                updated.add("Y99")
            elif line.startswith("- Current observer="):
                line = "- Current observer=y99_deployment_v2.py observe --id AdaRES-Y99|AdaRES-Y98 --token CURRENT --label NEW. Older observers reject HEADY99. Never live-observe/rearchive Y96/Y95/Y94 or earlier archived runs."
                updated.add("observer")
            elif line.startswith("- AdaRES-Y96 original PID"):
                continue
            elif line.startswith("- AdaRES-Y98 original PID"):
                row = observations["AdaRES-Y98"]
                p = row["phase"]
                line = "- AdaRES-Y98 original PID25962/server2GPU0/280W, " + p["phase"] + str(p["epoch"]) + "/30 step" + str(p["step"]) + "/" + str(p["count"]) + "; exact source/YAML/private runtime/power and native counter increase " + evidence(row["evidence"]) + "; commit" + Y98 + "."
                updated.add("Y98")
            elif line.startswith("- Y96 exact original directory"):
                line = "- Y96 exact original20261110_165702 naturally completed; full Train/Val1..30 stable finite last30/best23/model/optimizer544/scheduler30-23 verified ONCE and canonical archive " + evidence(y96["terminal_evidence"]) + "; trainingbest.5081601738929749 is NOT formal FIRST. Original full server1/350W frozen review now waits onY99; no restart/migration/substitution or terminal replay."
                updated.add("terminal")
            elif line.startswith("- Y98 exact original directory"):
                line = "- Y98 exact original directory20261110_080627 (not launch210741); NEW actual-Y99-native compatible29+8 terminal plan " + evidence(y98["terminal_handoff_plan"]) + ". Future finalize_y98_fixed30_y99_v2.py ONLY at HEADY98/Y99 after natural30/full Train-Val1..30/exact absent original process/stable finite last-best-model-optimizer-scheduler and unchanged source/runtime/280W. No Y98 live checkpoint loaded."
                updated.add("handoff")
            elif line.startswith("- Fresh canonical occupation "):
                line = "- Fresh canonical occupation " + "; ".join(evidence(r) for r in occupations.values()) + "; both authorizedGPU0 busy/reserved by originalY98/newY99. All original GPU1 frozen FIRST/cost remain user-held, never moved/substituted."
                updated.add("occupation")
            elif line.startswith("- Next tick first observe "):
                line = "- Next tick observe Y98/Y99 and dispatch qualified waiting work. Do not replay Y96 terminal, Y99 native or old CPU/binding evidence. Prepare exact original Y99 terminal helpers by read-only actual-directory discovery; never infer checkpoint directory from launch name or load live weights. Y98 uses new tested finalize_y98_fixed30_y99_v2.py only after natural30; full original selected-device protocol mandatory."
                updated.add("next")
            lines.append(line)
        assert updated == {"token", "backup", "head", "Y99", "observer", "Y98", "terminal", "handoff", "occupation", "next"}
        row = observations["AdaRES-Y99"]
        p = row["phase"]
        at = lines.index("## Open Questions And Risks")
        lines[at:at] = ["- AdaRES-Y99 original PID" + str(row["pid"]) + "/server1GPU0/350W, " + p["phase"] + str(p["epoch"]) + "/30 step" + str(p["step"]) + "/" + str(p["count"]) + "; canonical launch " + evidence(dispatch_ref) + "; increasing native counters " + evidence(row["evidence"]) + "; exact original checkpoint directory must be discovered before future terminal preparation.",
            "- Untested first operational drafts and corrected explicit attribute references preserved " + evidence(drafts) + "; " + evidence(corrected) + ". Corrected helpers passed actual-evidence29+8+11 guards; no scientific/native failure, guard relaxation, CPU scientific replay or retraining is implied."]
        lines += ["## Last Tick",
            "- " + token + ": Y96 original natural fixed30 full checkpoint proof/canonical archive; Y99 FIRST actual native PASS, originalY98 actual-native compatible terminal29+8, exact source selection,11 admission guards/full original server1 freeze/receipt/canonical dispatch/increasing counters verified. Original Y98 continues unchanged.",
            "- New scientificIDs0/scientificCPUreplays0/nativeGPUtasks1/admissions1/launches1/archives1/FIRST0/cost0. Goal/old receipts/data/runtime/mappings/seed1999fixed30/350W280W/bans/GPU1+Y57 pauses/foreign jobs intact. Closeout " + evidence(ref(output)) + "; official finish/receipt/release authoritative."]
        assert len(lines) < 120 and sum(r.startswith("## ") for r in lines) == 7
        memory.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        phases = [cid + " " + r["phase"]["phase"] + str(r["phase"]["epoch"]) + "/30 step" + str(r["phase"]["step"]) + "/" + str(r["phase"]["count"]) for cid, r in observations.items()]
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "运行：" + "；".join(phases) + "。原Y98进程不变，Y99在server-1 GPU0/350W按原生YAML固定30轮运行，精确源码/配置/私有环境/功耗和原生步数递增已验证。",
            "完成：Y96自然固定30轮结束，原目录20261110_165702完整Train/Val1..30和稳定有限last30/best23/model/optimizer544/scheduler30-23已核验并规范归档，原产物保留；训练best.508160不是正式FIRST。",
            "新准备并推进：Y99已有七个科学文件和CPU证据不重跑；首次真实TRAIN两步检查通过，loss[4.615511417388916,6.3237128257751465]，新路径/backward/更新和预算通过。原Y98精确终态兼容29+8项检查通过，11项实际证据准入检查通过，冻结完整原server-1协议、收据和规范派发完成。",
            "保留未测试操作原稿，明确版本属性修正另存，修正后的工具通过上述实际证据检查。本轮新科学ID0、CPU科学重跑0、GPU原生1、准入1、新训练1、归档1；无正式FIRST、成本评测或晋升。"
        ], decisions=[
            "保留原Y98、新Y99固定30轮运行及全部产物/收据/精确数据/私有环境/路径映射/seed1999和科学禁用项。Y96完整原设备冻结后评等待server-1空闲，不从训练日志推断效果。下一轮准备Y99精确终态工具，按实际原目录发现并绑定，不推断目录或加载活跃权重。",
            "server-2 GPU1全部使用与Y57单独暂停保持；不迁移冻结后评/成本，不重启受保护实验，不使用server6、不跨项目、不改调度或功耗、不干预外部任务。无新增用户处理事项。"
        ], evidence=[ref(output), ref(backup)]))
        assert all(ref(p) == entry for p, entry in before.items())
        assert git("ls-files", "--stage", "-z") == index
        return output, notes, len(lines)
    finally:
        writer.close()


def finish(token):
    lease(token, 240)
    for _, cid, _ in RUNS:
        command(["scripts/y99_deployment_v2.py", "observe", "--id", cid, "--token", token, "--label", "release"])
    for name, _, _ in RUNS:
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
        observations={cid: row["phase"] for cid, row in tick.load(output)["observations"].items()},
        native_GPU_tasks=1, admissions=1, launches=1, archives=1)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True)
    finish(p.parse_args().token)
