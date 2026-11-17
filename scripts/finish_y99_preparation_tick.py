"""Close FIRST Y99 CPU preparation with fresh physical occupation and live runs."""
import argparse
import ast
import base64
from datetime import datetime
import json
import os

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y94_deployment import Y93, Y94, contract, git
from prepare_y99_spatial_mask_skip import files
from y99_contract import validate_cpu
from run_y99_native import runtime, validate_binding
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


def evidence(entry):
    return entry["path"] + " SHA" + entry["sha256"]


def memory(token, view, new, observations, occupations, output):
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
                    + "; complete historical snapshot chain/bans/negative evidence/dedicated restoration "
                    "requirements byte-preserved, never permission to reopen.")
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
            line += (" Y99 follows Y95/Y96/Y98; same-byte source/body runtime is prepared ONLY at "
                     "HEADY94/currentY94-server1 orY93-server2 owners. Changed HEAD/owner requires "
                     "NEW tested binding, never replay CPU/version/plan.")
        lines.append(line)
    result = tick.load(closure.artifact(ROOT, new["CPU_result"]))
    where = lines.index("## Active Hypotheses") + 1
    lines[where:where] = [
        "- Y99 independent W2 full sparse-input mask-MLP bypass: mask_static=x_mask(x)+x@zeroW32x256 "
        "once before SAME original batching/seven dots; +8192/total164637829/no RNG. Complete x_mask and "
        "input_proj(x), full1024 appearance/input_proj_2d/sum_norm, all32 sparse channels/all points/full "
        "inputs/query norms/scores/shared indicator/refinement/sampling/SCG/gate algorithm/targets-loss-"
        "selector/releasedbackbone/seed1999/fixed30/native YAML unchanged. Structural connection, not "
        "input/feature subset/threshold/temperature/loss/epoch/validation scan.",
        "- Y99 not Y98 raw appearance-to-mask path, Y72 recurrent refined-visual masks, Y81 query-attended "
        "masks, Y85 mask norm, Y88/Y92 squared readouts or Y96 sampled-query skip. Same original spatial "
        "input already represented by full mask MLP; no missing information, defect, novelty, benefit, "
        "combination or rejected-route reopening claim. Earlier prepared candidates keep priority.",
        "- Y99 FIRST actual original CPU YAML/fullmodel/releasedbackbone/TRAIN162738points8expressions "
        "passed float32/64 independent scalar/no-RNG/row-channel permutations/zero/sign/row-local/both-"
        "input gradients, exact seven-stage W2/RNG/weight and close input-gradient anchors, causal "
        "initial/final masks/exact initial scores/bypass, full32 identity/both original projections/"
        "static seven consumers and two finite synthetic updates with387 gradient tensors. Learned "
        "mask-feature delta " + str(result["CPU_spatial_mask_paths"][1]["mask_feature_delta_max"])
        + "; CPU result " + evidence(new["CPU_result"]) + "; program " + evidence(new["CPU_program"])
        + ". NOT actual sparse GPU/native/admission/efficacy or full-data memory proof.",
        "- Y99 retained ONLY seven tested A files commit" + new["code_commit"] + "/" + new["retained_branch"]
        + " directly on actual-nativeHEADY94; main HEAD/index/queue/Y93-Y94 runs/earlierY95-Y96-Y98 unchanged. "
        "Version " + evidence(new["source_manifest"]) + "; native preparation "
        + evidence(new["native_preparation"])
        + ". Ten actual-evidence plus8 runtime guards passed FIRST execution/no skips; same native "
        "callbacks actually executed on CPU, never GPU.",
        "- Y99 runner=run_y99_native.py ONLY at HEADY94 after matching original Y94/server1 orY93/server2 "
        "natural30/full finite exact original checkpoint/canonical archive plus fresh exclusive idle/"
        "unreserved authorizedGPU0; earlierY95/Y96/Y98 precedeY99. Changed HEAD/owner requires NEW tested "
        "same-byte binding. Actual native PASS/full original freeze/receipt/admission/exact source/"
        "safe sync/increasing counters mandatory, no CPU/version/plan/terminal replay.",
    ]
    lines += [
        "## Last Tick",
        "- " + token + ": original Y93/Y94 PID/source/YAML/receipts/private runtimes/280W-350W/native "
        "counters verified; both authorizedGPU0 busy/reserved. No terminal/checkpoint loading/stop/"
        "restart/migration/archive or unauthorized GPU test.",
        "- Implemented ONE stable independent Y99 full32 sparse-input mask-MLP bypass after targeted "
        "history review. FIRST actual original CPU YAML/fullmodel/releasedbackbone/TRAIN load and "
        "component/anchors/causal-bypass/same-callback/two synthetic updates passed; ten evidence "
        "and8 runtime guards passed. No original scientific control relaxed or failure retried.",
        "- Retained only Y99 seven tested additions on actual-nativeY94 with isolated index; main "
        "HEAD/index/working files/current runs/earlier preparations preserved. Exact server1/server2 "
        "private runtimes/data/mappings/350W-280W/native plans ready, GPU unexecuted/unadmitted.",
        "- New scientificIDs1/scientificCPUattempts1/nativeGPUtasks0/admissions0/launches0/archives0. "
        "Y95 current binding/pre-admission and Y96/Y98 originals retained; Y97 original/revision1 numerical "
        "failures unchanged. No filler second candidate, new permission request or performance claim.",
        "- Goal bytes/old receipts/full9508x141 FIRST/two repeats/bootstrap10000 seed1999/TRAIN1487-3325/"
        "all harm metrics/original-device cost/data/runtime/fixed30/bans/dedicated restore/power/scheduler "
        "intact. GPU1 all use and Y57 separately paused; no server6/cross-project/foreign-job changes.",
        "- Closeout " + evidence(ref(output))
        + "; exact memory backup, fresh sequential native observations and canonical occupation, "
        "collector actual resource_check retained. Official finish requires "
        "FINALIZED/PASS/context_committed/released/absent owned lock/pending archive journal.",
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
        new = candidates["AdaRES-Y99"]
        assert new["stage"] == "blocked" and new["blocker_kind"] == "exclusive_gpu_slot"
        assert not new.get("native_smoke_result") and not new.get("admission_receipt")
        assert not (ROOT / "pipeline/admissions/AdaRES-Y99.json").exists()
        cpu = tick.load(closure.artifact(ROOT, new["CPU_result"]))
        validate_cpu(cpu)
        program = closure.artifact(ROOT, new["CPU_program"])
        payload = ast.literal_eval(ast.parse(program.read_text(encoding="utf-8")).body[0].value)
        assert cpu["goal_sha256"] == payload["goal_sha256"] == GOAL
        for path in files():
            value = payload["variant"]["config_data"] if path.suffix == ".yaml" else payload["variant"]["python_files"][path.name]
            assert path.read_bytes() == base64.b64decode(value["base64"]) and ref(path)["sha256"] == value["sha256"]
            assert path.read_bytes() == git("show", new["code_commit"] + ":" + path.relative_to(ROOT).as_posix())
        version = tick.load(closure.artifact(ROOT, new["source_manifest"]))
        validate_binding(version, new, Y94)
        assert git("rev-parse", new["code_commit"] + "^").decode().strip() == Y94
        assert len(version["exact_delta"]) == 7 and all(status == "A" for status, _ in version["exact_delta"])
        preparation = tick.load(closure.artifact(ROOT, new["native_preparation"]))
        assert preparation["token"] == token and preparation["guard_tests_passed"] == 10
        assert preparation["runtime_guard_tests_passed"] == 8
        assert preparation["same_callbacks_actually_executed_on_CPU"] is True
        assert preparation["native_observer_callbacks_executed_on_GPU"] is False
        assert preparation["native_executed"] is False and preparation["admitted"] is False
        for entry in preparation["evidence"]:
            closure.artifact(ROOT, entry)
        for name, key in (("server-1", "native_smoke_plan_server1"), ("server-2", "native_smoke_plan_server2_gpu0")):
            plan = tick.load(closure.artifact(ROOT, new[key]))
            source = closure.artifact(ROOT, plan["program"]).read_text(encoding="utf-8")
            frozen = ast.literal_eval(ast.parse(source).body[0].value)
            assert validate_frozen(new, plan, frozen) == 7
            assert runtime(name, new, frozen, servers[name]) == (350. if name == "server-1" else 280.)
        for entry, count in ((preparation["guard_tests"], 10), (preparation["runtime_guard_tests"], 8)):
            raw = closure.artifact(ROOT, entry).read_bytes()
            assert ("Ran " + str(count) + " tests").encode() in raw and b"OK" in raw and b"skipped" not in raw
        for cid in ("AdaRES-Y95", "AdaRES-Y96", "AdaRES-Y98"):
            old = candidates[cid]
            assert old["stage"] == "blocked" and not old.get("native_smoke_result")
            assert not old.get("admission_receipt")
            for key in ("CPU_result", "CPU_program", "source_manifest", "native_preparation",
                        "native_smoke_plan_server1", "native_smoke_plan_server2_gpu0"):
                closure.artifact(ROOT, old[key])
        numerical = candidates["AdaRES-Y97"]
        assert numerical["stage"] == "blocked" and numerical["blocker_kind"] == "numerical_control_failure"
        assert len(numerical["revision_history"]) == 1 and numerical["revision_history"][0]["result"] == "failed"
        closure.validate_revisions(ROOT, numerical)
        output = ROOT / ("pipeline/checks/y99_preparation_closeout_" + token + ".json")
        assert not output.exists()
        dependencies = [ref(__file__), new["CPU_result"], new["CPU_program"], new["source_manifest"],
            new["native_preparation"], numerical["CPU_failure"], numerical["revision1_result"]] + [
                entry["evidence"] for entry in observations.values()] + list(occupations.values())
        tick.save(output, dict(
            complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            FIRST_actual_original_Y99_CPU_scientific_attempt_passed=True,
            same_callbacks_CPU_executed_and_native_GPU_unexecuted=True,
            isolated_seven_tested_additions_retained=True, actual_evidence_guards=10, runtime_guards=8,
            original_Y93_Y94_PID_source_training_yaml_receipts_full_freezes_preserved=True,
            earlier_Y95_Y96_Y98_and_Y97_original_revision1_failures_preserved=True,
            original_training_checkpoints_loaded=False, new_scientific_candidates=1,
            scientific_CPU_candidate_attempts=1, native_GPU_tasks=0,
            new_admissions=0, new_training_launches=0, exact_original_archives=0,
            production_HEAD_index_queue_preserved=True, full_original_FIRST_and_cost_preserved=True,
            original_GPU1_and_Y57_separate_pause_preserved=True,
            observations=observations, occupations=occupations, efficacy_measured=False, evidence=dependencies))
        new["evidence"].append(ref(output))
        assert work["tick_attempt"]["kind"] == "resource_check"
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, lines = memory(token, view, new, observations, occupations, output)
        decision = monitor.log_decision(str(ROOT), "next_batch",
            "GPU0满载且无ready waiting，如何准备一个与已有掩码路线可区分的独立变体？",
            "独立W2保留完整x_mask，增加全部原32维稀疏输入的零初始化线性旁路；稳定ID Y99。"
            "首次实际CPU构造/原TRAIN加载、独立数值/七阶段锚点/梯度/因果旁路/两步合成更新通过，"
            "仅留版七文件并准备原生检查，GPU未执行且不入队。",
            "Y98是原1024维外观到掩码的新路，Y99是原掩码MLP同一输入的并行参数化；"
            "Y72/Y81是当前精炼或query-attended状态，Y92平方已投影掩码。无新输入、删减或选择"
            "特征、无组合/重开/调参/收益预判。10证据及8runtime guard通过；两GPU0忙，不抢占。",
            output.relative_to(ROOT).as_posix())
        phases = []
        for cid, entry in observations.items():
            phase = entry["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30，步"
                          + str(phase["step"]) + "/" + str(phase["count"]))
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅AdaRES_project。核验实际owner、Disabled后台调度、正式skill/experiment_first、"
            "AGENTS/goal/SERVERS/最新GPU1暂停后begin；先查询进展及派发现有合格队列，ready waiting为0。",
            "运行：" + "；".join(phases) + "。原PID/source/YAML/收据/私有runtime/原数据、"
            "350W/280W与原生步数递增确认；canonical只读物理占用两个授权GPU0均busy+reserved。"
            "没有抢占、GPU1使用、live checkpoint加载、stop/restart/migration/新终态归档。",
            "新准备：独立W2变体Y99，保留完整原静态掩码MLP并添加全部32维原稀疏输入的线性"
            "零初始化旁路，一次、+8192参数/no RNG、七阶段共用。全部原空间/外观输入及"
            "投影、坐标/采样/SCG、六次精炼/共享confidence/原损失和gate算法保留。"
            "与Y98外观直达、Y72循环visual、Y81 query-attended、Y92平方已投影特征不同；"
            "未做新数据/特征删减选择/阈值或验证集扫参、组合或重开旧拒绝。",
            "Y99首次实际CPU执行通过：原YAML/fullmodel/releasedbackbone/TRAIN162738点8表达，"
            "float32/64独立scalar/no-RNG/行及输入输出通道置换/零/符号/局部干预/双输入梯度，"
            "七阶段W2权重/RNG/输出精确锚点与输入梯度锚点，初始和最终mask干预、初始scores"
            "不变及exact bypass，完整原32维身份/两原visual投影/原batching/静态七consumer。"
            "两步合成更新有限，各387梯度张量，更新后mask-feature差0.002486199140548706。",
            "仅七项已测新增文件保留在实际native通过的HEADY94父版本，使用独立index，"
            "主HEAD/index/现有训练/较早Y95-Y96-Y98原工件保持。10项实际证据和8项runtime "
            "guard首次通过；server1/server2独立原生YAML/私有环境/350W-280W/精确数据映射计划就绪。"
            "同一native callback已实际在CPU执行，GPU未执行，不冒充native PASS/准入或full-data memory。",
            "本轮新科学ID1、科学CPU尝试1、GPU原生0、准入0、启动0、归档0。Y95执行绑定/"
            "server2准入前检查、Y96/Y98以及Y97原始和修订1数值失败均保持。精确备份七节记忆、"
            "刷新原训练及canonical占用、保留collector实际resource_check，再正式report/finish，不预称收尾结果。"
        ], decisions=[
            "不强凑第二候选或伪造waiting。Y99排在Y95/Y96/Y98之后；仅当前HEADY94、"
            "匹配原Y94/server1或Y93/server2自然30轮完整有限原检查点核验/canonical归档、"
            "实时独占空闲GPU0后执行run_y99_native.py。未来HEAD或owner变化须新同字节"
            "测试绑定，不重放CPU/version/plan；实际native通过再当轮完整freeze/receipt/"
            "准入/精确source/安全派发/递增counter。",
            "科学goal字节、原签名收据、原数据/私有runtime/mappings/seed1999固定30轮、"
            "禁用和专用恢复/功耗/调度保持。完整9508表达141场景/candidate+Y36-W2-B1各两次/"
            "FIRST/场景bootstrap10000-seed1999/TRAIN1487-3325/全部有害子组和指标/原设备cost不变。"
            "GPU1全部使用与原GPU1冻结后评继续暂停，Y57独立暂停；禁止server6/跨项目/外部任务干预。"
            "CPU可行性不是训练收益，正式效果待原冻结评测，无新用户操作需求。"
        ], evidence=[ref(output), backup, ref(decision)]))
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
        new_candidates=1, scientific_CPU_attempts=1, new_admissions=0, launches=0, archives=0),
        ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    finish(parser.parse_args().token)
