"""Persist exact Y96 preparation and close the lease with fresh physical evidence."""
import argparse
import ast
from datetime import datetime
import json
import os
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y93_deployment import Y92, Y93, contract, git
from y96_contract import validate_cpu
from run_y96_native import runtime, validate_binding
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


def update_memory(token, view, candidates, observations, occupation, output):
    path = ROOT / "pipeline/codex_memory.md"
    backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
    assert not backup.exists()
    original = path.read_bytes()
    backup.write_bytes(original)
    assert backup.read_bytes() == original
    old = original.decode("utf-8").splitlines()
    lines = old[:old.index("## Last Tick")]
    new = candidates["AdaRES-Y96"]

    def evidence(value):
        return value["path"] + " SHA" + value["sha256"]

    for index, line in enumerate(lines):
        if line.startswith("- Current token "):
            lines[index] = "- Current token " + token + "; snapshot " + view["generated_at"] + "; deadline " + view["deadline_at"] + "; never extend."
        elif line.startswith("- Exact previous seven-section memory:"):
            lines[index] = "- Exact previous seven-section memory: " + evidence(ref(backup)) + "; all earlier snapshot-chain history/bans/negative evidence/dedicated recovery instructions byte-preserved. Compression never authorizes reopening."
        elif line.startswith("- Current AdaRES-Y92 ") or line.startswith("- Current AdaRES-Y93 "):
            cid = line.split()[2]
            entry = observations[cid]
            phase = entry["phase"]
            lines[index] = "- Current " + cid + " " + phase["phase"] + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"]) + "; original native progress/source/YAML/runtime/power verified: " + evidence(entry["evidence"])
        elif line.startswith("- Fresh canonical occupation:"):
            lines[index] = "- Fresh canonical occupation: " + "; ".join(evidence(value) for value in occupation.values()) + "; both authorized GPU0 busy/reserved; age<300s required. Keep collector's actual tick_attempt and all candidate evidence."
        elif line.startswith("- Next tick first observeY92/Y93"):
            lines[index] = "- Next tick first observeY92/Y93 and dispatch ready work. Three GPU-unexecuted/unadmitted prepared independent candidates Y94 thenY95 thenY96. Only matching exact original-owner full30/checkpoint/archive and fresh idle authorized GPU0 permits native execution. No CPU/retain/plan or archived-terminal replay; changed HEAD/owner requires NEW tested terminal/observer/candidate bindings first."
    where = lines.index("## Active Hypotheses") + 1
    lines[where:where] = [
        "- Y96 independent W2 sampled-seed initial-query skip: q_init=original_query_generator(sampled_seed)+sampled_seed@zeroW256x256, once; +65536/total164695173/no RNG. Same undetached post-SCG sampled128x256 seed and all five original generator layers retained. Original sampling/positions/static masks/six refinements/SCG hard neighborhoods/coarse-fine average/self-language sum/shared heads/fullinputs/2D/backbone/targets-loss-selector/seed1999/fixed30/nativeYAML unchanged.",
        "- Y96 not repeated generated-query anchoringY73, pooled-languageY75, coordinatesY77, global visual meanY84, query-set recurrent meanY95 or confidence evidence skipY93. No combination/reopening/missing-information/defect/novelty/benefit claim. Original final-confidence/data-premise/gate/norm rejections and Y57 pause untouched.",
        "- Y96 FIRST actual CPU native YAML/fullmodel/releasedbackbone/TRAIN162738points8expressions passed: float32/64 independent scalar/no-RNG/slot-batch-channel permutation/zero-input/slot-local intervention/both-input gradients, exact seven-stage W2/RNG/weights and close input-gradient anchors, causal initial/final-mask intervention/exact bypass and two finite synthetic updates with387 gradient tensors; new skip delta after first update .013011366128921509. CPU result " + evidence(new["CPU_result"]) + "; CPU program " + evidence(new["CPU_program"]) + ". NOT actual sparse GPU task/admission/efficacy.",
        "- Y96 retained ONLY seven tested A files, commit" + new["code_commit"] + "/refs/heads/adares-y96-seed-query-skip directly on actually native-trained HEADY93. Version " + evidence(new["source_manifest"]) + "; production HEAD/index/queue/earlierY94/Y95 preserved. Ten actual-evidence and8 runtime guards passed FIRST execution/no skip; preparation " + evidence(new["native_preparation"]) + ". Same original SCG sampled-seed/generator/initial-coarse/head/gate callbacks actually executed on CPU; GPU callbacks unexecuted.",
        "- Y96 runner=run_y96_native.py ONLY at HEADY93 after matching Y92/server1 orY93/server2 natural full30/stable finite exact checkpoint/canonical archive and fresh idle/unreserved authorized GPU0. Y94/Y95 precedeY96; changed HEAD/owner requires NEW tested binding. Never replay CPU/version/plan. Actual native PASS/full original freeze/receipt/admission/exact source selection/safe sync/increasing native counters still required.",
    ]
    lines += [
        "## Last Tick",
        "- " + token + ": original Y92/Y93 native progress/source/YAML/receipt/runtime/350W-280W verified; no new terminal. Existing exact Y92 current30+8 and Y93 original29 terminal plans only hash-checked, never executed or checkpoint-loaded. Original Y94/Y95 preparatory evidence preserved without scientific replay.",
        "- New independentY96 actual original CPU/model/backbone/TRAIN, scalar/slot-local seed controls, seven-stage W2/input-gradient anchors/initial-final intervention/exact bypass and two synthetic updates passed FIRST execution. Seven exact A files retained using private index directly on testedY93;10 evidence+8 runtime guards passed with no skip. No new scientific failure/revision.",
        "- New scientific candidates1/actual CPU candidate runs1/GPU native tasks0/admissions0/launches0/archives0. All original goal bytes/receipts/complete frozen FIRST and same-device cost/data/runtime/mappings/seed1999-fixed30/bans/current power/schedule and restore rules unchanged. GPU1 all use/Y57 separate experiment remain paused; no user action required.",
        "- Closeout " + evidence(ref(output)) + "; sequential fresh native observations/canonical occupation, keep collector's actual resource_check, then official report/finish. No preclaim; afterward verify FINALIZED/PASS/context_committed/released and absent lock/pending/archive journals read-only. No post-release writes.",
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
        observations, occupation = {}, {}
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
            occupation[name] = proof_ref
            terminal = tick.load(closure.artifact(ROOT, candidates[cid]["terminal_handoff_plan"]))
            assert terminal["checkpoint_loaded"] is False and terminal["archived"] is False
            assert terminal["post_training_freeze"] == candidates[cid]["post_training_freeze"]
        for cid in ("AdaRES-Y94", "AdaRES-Y95", "AdaRES-Y96"):
            candidate = candidates[cid]
            assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
            assert not candidate.get("native_smoke_result") and not candidate.get("admission_receipt")
            assert not (ROOT / ("pipeline/admissions/" + cid + ".json")).exists()
        assert candidates["AdaRES-Y94"]["source_manifest"]["sha256"] == "437d51952a3702ad341771a09c85b371a2eba9056bf54ab446283f33a04cfab4"
        assert candidates["AdaRES-Y95"]["source_manifest"]["sha256"] == "49091dda2a519022d63df8e4c3cee9b61b732bc08c75b08faa684f938eaee974"
        for cid in ("AdaRES-Y94", "AdaRES-Y95"):
            for key in ("CPU_result", "source_manifest", "native_preparation"):
                closure.artifact(ROOT, candidates[cid][key])
        new = candidates["AdaRES-Y96"]
        validate_cpu(tick.load(closure.artifact(ROOT, new["CPU_result"])))
        version = tick.load(closure.artifact(ROOT, new["source_manifest"]))
        validate_binding(version, new, Y93)
        assert version["CPU_result"] == new["CPU_result"]
        preparation = tick.load(closure.artifact(ROOT, new["native_preparation"]))
        assert preparation["guard_tests_passed"] == 10 and preparation["runtime_guard_tests_passed"] == 8
        assert preparation["same_callbacks_actually_executed_on_CPU"] is True and preparation["native_executed"] is False
        for key, count in (("guard_tests", 10), ("runtime_guard_tests", 8)):
            raw = closure.artifact(ROOT, preparation[key]).read_bytes()
            assert ("Ran " + str(count) + " tests").encode() in raw and b"\nOK" in raw and b"skipped" not in raw
        for name, key in (("server-1", "native_smoke_plan_server1"), ("server-2", "native_smoke_plan_server2_gpu0")):
            plan = tick.load(closure.artifact(ROOT, new[key]))
            source = closure.artifact(ROOT, plan["program"]).read_text(encoding="utf-8")
            frozen = ast.literal_eval(ast.parse(source).body[0].value)
            assert validate_frozen(new, plan, frozen) == 7
            runtime(name, new, frozen, servers[name])
        output = ROOT / ("pipeline/checks/y96_preparation_closeout_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, token=token, recorded_at=datetime.now().astimezone().isoformat(),
            observations=observations, actual_occupation=occupation, Y96_CPU_result=new["CPU_result"],
            Y96_retained_version=new["source_manifest"], Y96_native_preparation=new["native_preparation"],
            earlier_preparations={cid: candidates[cid]["source_manifest"] for cid in ("AdaRES-Y94", "AdaRES-Y95")},
            CPU_guards=10, runtime_guards=8, original_Y92_Y93_and_earlier_Y94_Y95_preserved=True,
            original_full_frozen_FIRST_and_cost_preserved=True, original_GPU1_and_Y57_pause_preserved=True,
            new_scientific_candidates=1, actual_new_CPU_candidate_runs=1, actual_GPU_native_tasks=0,
            new_admissions=0, new_training_launches=0, exact_original_archives=0,
            earlier_CPU_scientific_replays=0, original_training_checkpoints_loaded=False,
            efficacy_measured=False, production_HEAD_index_queue_preserved=True,
            evidence=[ref(__file__), new["CPU_result"], new["CPU_program"], new["source_manifest"],
                new["native_preparation"]] + [row["evidence"] for row in observations.values()] + list(occupation.values())))
        new["evidence"].append(ref(output))
        assert work["tick_attempt"]["kind"] == "resource_check"
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, lines = update_memory(token, view, candidates, observations, occupation, output)
        decision = monitor.log_decision(str(ROOT), "next_batch",
            "两授权GPU0仍被原Y92/Y93占用，本轮快照允许补充独立准备。",
            "新增Y96原采样种子到初始查询生成器的线性旁路；原CPU实测及10+8守卫通过后独立留版，Y94/Y95保持优先。",
            "原始YAML/完整模型/发布骨干/TRAIN真实加载、独立标量和slot-local因果、七阶段W2与输入梯度、精确旁路及两步合成更新通过；GPU原生任务及训练收益仍未测。",
            output.relative_to(ROOT).as_posix())
        phases = []
        for cid, entry in observations.items():
            phase = entry["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30，步"
                + str(phase["step"]) + "/" + str(phase["count"]))
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅处理AdaRES_project；实际owner会话、Disabled后台调度、正式前台skill/experiment_first与当前AGENTS/goal/SERVERS/GPU1暂停记录已核验。begin先查进展和已有合格派发；初始ready waiting为0，无新终态或操作故障。",
            "运行：" + "；".join(phases) + "。原PID/source/YAML/receipt/私有runtime及350W/280W、原生步数递增通过；canonical物理占用确认两GPU0 busy+reserved。未使用GPU1，不干扰外部任务。",
            "新准备：独立Y96只在原查询生成器后，将相同未detach的post-SCG采样种子经zeroW256x256投影一次加入初始查询，保留完整原五层生成器、采样/位置、静态mask及六级原细化/硬门/SCG/共享head等原协议。",
            "首次真实原YAML/完整CPU模型/发布骨干/TRAIN162738点8表达，float32/64独立标量/no-RNG/slot-batch-channel置换/零输入/逐slot种子干预及双输入梯度、七阶段W2/RNG/权重和输入梯度锚点、初始与最终mask因果干预/精确bypass、两步有限合成更新通过。相同未来native回调实际在CPU执行，核验原SCG采样种子、五层生成器、初始预测和首轮coarse消费者。不是稀疏GPU任务或收益证据。",
            "仅七个测试新增文件经私有index留在Y93原生已测父版本的独立分支；生产HEAD/index/队列/无关改动及较早Y94/Y95保持。10项实际证据和8项runtime/goal/source/owner拒绝守卫全部通过，无跳过，保存两授权GPU0计划和精确runner。未GPU检查/准入/同步/派发。",
            "原Y92/Y93精确终态计划仅只读核对哈希，未重放生命周期测试或加载checkpoint；已归档Y91及更早实验不重启、不重放。Y94/Y95未重放CPU/原生/留版检查，仍待对应原owner自然fixed30完整归档和授权实时空槽。",
            "本轮新增科学候选1、CPU候选实测1、GPU原生任务0、准入0、启动0、归档0。七节记忆先精确字节备份再更新；收尾连续刷新原生训练和canonical物理占用，保留collector实际resource_check，再正式report/finish，不预先宣称postflight通过。"
        ], decisions=[
            "原Y92/Y93继续；Y94、Y95先于Y96。只有匹配原GPU0 owner自然30轮/完整有限checkpoint精确核验归档并实时独占空闲，才执行真实原生两步，通过后按原完整freeze/admission/安全同步和递增counter派发。改变HEAD或owner须新已测绑定，不以CPU或旧归档替代。",
            "原9508表达/141场景、候选+Y36/W2/B1各两次、FIRST主结果、bootstrap10000/seed1999、TRAIN cuts1487/3325、有害子组、全部原指标和原同设备cost保持；不声称新收益、晋升或达标。",
            "server2 GPU1全部使用及原GPU1后评/cost继续暂停且不迁移替代；Y57另行暂停，各自恢复需用户明确授权。禁止server6，不改变科学goal字节、原运行receipt、数据/环境/mappings/seed/fixed30/gates/功率/调度或专用恢复要求。无需新用户处理。"
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
