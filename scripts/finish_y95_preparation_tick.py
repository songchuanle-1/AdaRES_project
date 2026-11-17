"""Close the owned tick using fresh sequential observations and immutable evidence."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from y93_deployment import Y92, Y93, contract, git
from y95_contract import validate_cpu
from run_y95_native import runtime, validate_binding
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


GOAL = "4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a"
PAUSE = "3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45"


def fresh(value, key):
    age = (datetime.now().astimezone() - datetime.fromisoformat(value[key])).total_seconds()
    assert 0 <= age < 300, "Refresh the actual observation under a new immutable label"


def memory(token, view, candidates, observations, occupations, closeout):
    path = ROOT / "pipeline/codex_memory.md"
    backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
    assert not backup.exists()
    original = path.read_bytes()
    backup.write_bytes(original)
    assert backup.read_bytes() == original
    old = original.decode("utf-8").splitlines()
    assert sum(line.startswith("## ") for line in old) == 7
    y92, y93, y94, y95 = (candidates["AdaRES-" + name] for name in ("Y92", "Y93", "Y94", "Y95"))

    def evidence(value):
        return value["path"] + " SHA" + value["sha256"]

    lines = [
        "# AdaRES_project Research Memory",
        "## Goal And Phase",
        "- Current token " + token + "; snapshot " + view["generated_at"] + "; deadline " + view["deadline_at"] + "; never extend.",
        "- Actual CODEX_THREAD_ID=owner01a07e32-78bf-7670-925b-370911ea48db; Pipeline-AdaRES_project Disabled; experiment_first enabled.",
        "- Only server1GPU0 existing350W and server2GPU0 fixed280W, one experiment/card; all server2GPU1 use user-paused, Y57 separately paused; no server6/cross-project/scheduler/power changes.",
        "- Resource authority pipeline/authorizations/server2_gpu1_pause_20261104.json SHA" + PAUSE + "; overrides oldGPU1 grant. Scientific goal SHA" + GOAL + " remains byte-identical; original signed receipts preserved.",
        "- PROJECT_ROOT=D:/project; PYTHONDONTWRITEBYTECODE=1; PYTHONUTF8=1; python -B. SSH keys work; no private password file/sudo assumptions.",
        "- Exact previous seven-section memory: " + evidence(ref(backup)) + "; full snapshot chain retains ALL history, failed probes, negative evidence, bans and dedicated recovery requirements. Compression is not deletion or reopening.",
        "## Champion And Baseline",
        "- Target mIoU .5150. Frozen formal FIRST Y36=.510725438294, W2=.510107995189, B1=.502221764356, Y52=.509708428366, Y54=.506342952986; historical CSV Y36 .5110 is not replacement FIRST.",
        "- Full original review:9508 expressions/141 scenes, candidate+Y36/W2/B1 each twice, FIRST primary, scene bootstrap10000/seed1999, TRAIN cuts1487/3325, all subgroup harm/mIoU/Acc25/50/activequeries/latency/memory and original-device cost.",
        "- No formal FIRST/cost/promotion this tick. Y54 large-object harmCI[-.02678985,-.00070265], Y52 overallCI crosses0; single seed cannot estimate seed variance. Training best never replaces FIRST.",
        "## Active Hypotheses",
        "- Y95 independent W2 all-query mean bypass: fused=self_query+query_qla+mean_all128(query_qla)@zeroW256x256 before SAME original norm_qqa/FFN, six calls; +65536/total164695173/no RNG. Original local self attention/SCG hard neighborhoods/k/gamma remain untouched; the new global context is a separate path. Original full inputs/data/2D/backbone/sampling/static masks/coarse-fine average/head sharing/targets/loss/selector/seed1999/fixed30/native YAML unchanged.",
        "- Y95 not overlap-weighted recurrent feedbackY66, query-to-language/visual/mask memoriesY79/Y80/Y81, initial scene-visual meanY84, same-slot productY87 or self-attention geometryY94. No combination/reopening/missing-information/defect/novelty/benefit claim.",
        "- Y95 FIRST actual CPU execution passed original YAML/full model/released backbone/TRAIN162738points8expressions, float32/64 scalar/no-RNG/query-batch-channel permutations/replication/changed-peer unchanged-focal/zero-mean controls, exact seven-stage W2/RNG/weights and close input-gradient anchors, intervention/exact bypass and two finite synthetic updates with387 gradient tensors. CPU result " + evidence(y95["CPU_result"]) + "; CPU program " + evidence(y95["CPU_program"]) + ". CPU is NOT sparse GPU native/admission/efficacy.",
        "- Y95 retained seven tested A files commit" + y95["code_commit"] + "/refs/heads/adares-y95-query-set-mean directly on actually native-trained HEADY93; main HEAD/index and earlierY94 preserved. Version " + evidence(y95["source_manifest"]) + ". Ten actual-evidence plus8 runtime guards passed first attempt/no skip; native preparation " + evidence(y95["native_preparation"]) + ". Same observer callbacks actually executed on CPU; GPU callbacks unexecuted.",
        "- Y95 runner=run_y95_native.py ONLY at HEADY93 after matching Y92/server1 orY93/server2 natural30/full exact finite checkpoint/canonical archive and fresh idle/unreserved GPU0; earlierY94 has priority. Further HEAD/owner changes require NEW tested bindings. Actual native PASS/full original freeze/receipt/admission/source selection/safe sync/increasing counters still mandatory. No CPU/version/plan replay.",
        "- Y94 independent W2 shared bounded original-query-position self-attention bias weight[8,3]/+24/no RNG; six original query-self consumers, exact original SCG hard exclusions. Actual original CPU/scalar/custom-attention/anchors/input-gradient/bypass/two synthetic updates already passed in9cbbb; never replay. Detailed mechanism and all original proof hashes in exact backup.",
        "- Y94 CURRENT same-seven-file additive commit3f1c351f8a04e12745526e9abb3dbac46af806f1 directly onY93. Binding " + evidence(y94["additive_version_binding"]) + " passed10 guards. Only runner run_y94_bound_native_y93.py at HEADY93 after exact matching current-owner full30/archive/fresh idle GPU0. No GPU native/admission yet; old runners stale.",
        "- Y93 independent W2 full normalized evidence E769 to second confidence hidden h2: L3(h2+E@zeroW769x256), +196864/total164826501/no RNG. All original five MLP layers/norms/seven-stage sharing/evidence/masks/scores/protocol retained. Not Y91 hidden multiplicative gate orY86 scalar shortcut; no combination/reopening.",
        "- Y93 CPU/native/admission/selection/launch executed ONCE historically; never replay unused alternatives. Actual original server2GPU0/280W native losses[4.615511417388916,6.234131813049316], peak_reserved6704594944,13 operational+22 actual-native admission guards. Receipt " + evidence(ref("pipeline/admissions/AdaRES-Y93.json")) + "; full original freeze " + evidence(y93["post_training_freeze"]) + ". Freeze inherits complete originalY91 training/evaluation with method-ID only.",
        "- Y92 independent W2 static visual mask feature quadratic residual x_mask+x_mask.square()@zeroW256x256 once before original batching/seven dots; +65536/no RNG. Original queries/norm/score/indicator/refinement/all protocol retained. CPU/native/admission/launch executed ONCE, no replay. Actual native server1GPU0/350W losses[4.615511417388916,6.318481922149658], peak_reserved6717177856. Receipt " + evidence(ref("pipeline/admissions/AdaRES-Y92.json")) + "; full freeze " + evidence(y92["post_training_freeze"]) + ".",
        "## Closed Or Failed Directions",
        "- ALL earlier route definitions, closed/rejected scopes, source/CPU/native receipts, archives and failures are byte-preserved in the exact previous memory and its snapshot chain. Never reinterpret compression as authority to reopen. Existing AGENTS and project artifacts take precedence.",
        "- Preserve Y53/Y55/Y50/Y51/Y49rev1/Y46/Y48 and goal bans; F1 mask-entropy rejected. No validation scans of entropy/margin/occupancy/threshold/temperature/loss/features/epochs. Y22/Y23 and other dedicated restore requirements remain exact inAGENTS/backup.",
        "- Y86 original GPU0 native failed second-batch stage4 positive-response guard; coefficient about1e-4/overlap1.3414648492471315e-5/float32 delta0. Original guard/model/CPU/failure logs preserved, implementation numerical hold; not native PASS, no scientific rejection or revision. Never relax frozen numerical control.",
        "- Preserve prior observer/freshness/metadata/wrapper failures and narrow repairs: Y89 same-dtype reference correction at unchanged tolerance; Y78 CPU wrapper precision/optimizer-group fixes; Y80 162738-point metadata repair; Y88 helper boundary repair; Y94 rebind env wrapper failure/v2; all exact artifacts in backup. No new failure/repair this tick.",
        "- No archived fixed30 terminal helper or checkpoint loading replay. Preserve ALL original logs/checkpoints/freezes/receipts, original-device full review/cost requirements, server1 private torch/CUDA11.3 and server2 dedicated recovery rules.",
        "## Running And Queue",
        "- Production HEAD/native-trained identityY93=" + Y93 + "; observer y93_deployment.py observe --id AdaRES-Y92|AdaRES-Y93 --token CURRENT --label NEW; never old y92 observer atHEADY93.",
        "- Y92 original PID1305468/server1GPU0/350W/code" + Y92 + "; actual save directory20261108_090202 from native declaration. Future terminal ONLY finalize_y92_fixed30_y93.py with current30+8 binding " + evidence(y92["terminal_handoff_plan"]) + "; no checkpoint loaded/stopped/archived.",
        "- Y93 original PID47380/server2GPU0/280W/codeY93; original log train_adares_y93_evidence_hidden_skip_res_20261108_212628.log. NEW actual save directory20261108_082521 discovered from native save declaration, not launch timestamp. Future terminal finalize_y93_fixed30.py with29 lifecycle guards, exact source/YAML/receipt/freeze/location binding " + evidence(y93["terminal_handoff_plan"]) + "; no checkpoint loaded/stopped/archived.",
        "- Server1=songcl@192.168.1.175/dl02-MS-7B73; code=/data1/songcl/Project/AdaRES_project/AdaRES; privatePython=../runtime/adares_project/bin/python, LD_LIBRARY_PATH=../runtime/cuda113lib64, exactdata=../data/, only two existing SERVERS mappings. Server2=songcl@192.168.1.163/dl01-MS-7C60; code=/bak_home/songcl/Project/AdaRES_project/AdaRES; Python=/home/songcl/.conda/envs/adares_project/bin/python; exactdata=/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/.",
        "- Y57 separately user-paused at original epoch27/model/optimizer543/scheduler27; explicit Y57 continuation required for28..30. Generic GPU1 grant never permits restart/resume. ForeignGPU1 compute remains untouched.",
        "- Y81/Y82/Y83 and all earlier fixed30 archives retain original products; never restart/live-observe/re-finalize them. Y91 canonical full30 archive already completed in2683, originaldir20261107_102124, stable finite last30/best21/model/optimizer544/scheduler30-21, full TrainVal/source/YAML/receipt/freeze; trainingbest.5113725066184998 is NOT formalFIRST.",
        "- Y91 terminal SHA6b8df052be43d2ff03f46ab483bdf14892728c92e83788082877342a4770d916; archive SHAa6bf4009e2d1a605eb757ac73c20e87266f19107e46e50cb096931a8ba0d0bcc. Original GPU0/280W full frozen review waits onY93. EarlierY56/Y58..Y90 full original-device review remains as recorded; GPU1 routes stay user-held with no migration/substitution.",
    ]
    for cid, entry in observations.items():
        phase = entry["phase"]
        lines.append("- Current " + cid + " " + phase["phase"] + str(phase["epoch"]) + "/30 step"
            + str(phase["step"]) + "/" + str(phase["count"]) + "; native increasing/source/YAML/runtime/power verified: " + evidence(entry["evidence"]))
    lines += [
        "- Fresh canonical occupation: " + "; ".join(evidence(value) for value in occupations.values())
            + "; both authorized GPU0 busy/reserved, age<300s required. Preserve collector's actual resource_check tick_attempt and all candidate evidence.",
        "## Open Questions And Risks",
        "- No formal FIRST/cost or scientific promotion. New CPU finite losses/gradients and training best cannot prove task benefit/full-data memory. GPU1 all use and original GPU1 frozen review remain paused; Y57 separate pause remains.",
        "- Next tick first observeY92/Y93 and dispatch ready work. Two GPU-unexecuted/unadmitted prepared independent candidates Y94 thenY95. Only matching exact original-owner full30/archive plus fresh idle GPU0 permits native execution; no CPU/retain/plan replay, no old archived dependency substitution. Further HEAD changes require tested terminal/observer/pending-candidate bindings first.",
        "## Last Tick",
        "- " + token + ": Y93 exact terminal helpers prepared from actual native save declaration;29 lifecycle guards passed, no checkpoint loaded or training stopped. Y92 original30+8 binding and both running jobs preserved.",
        "- New independentY95 actual original CPU/model/backbone/TRAIN/scalar/set-peer/anchors/input-gradient/causal-bypass/two updates passed FIRST execution; exact seven-file isolated retention onHEADY93 and10+8 evidence/runtime guards passed. No prior CPU/native or terminal replay; no new scientific failure/revision.",
        "- New candidates1/actual CPU candidate runs1/GPU native tasks0/admissions0/launches0/archives0. Both GPU0 remain occupied. Goal/receipts/data/full original evaluation/cost/seed1999-fixed30/gates/power/schedule/restore requirements preserved. No new user action needed.",
        "- Closeout " + evidence(ref(closeout)) + "; sequential final native observations/physical occupation/report/finish. Before finish no postflight claim; afterward verify only FINALIZED/PASS/context_committed/released and absent lock/pending/archive journal. No post-release writes.",
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
        by_id = {row["id"]: row for row in work["candidates"]}
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
            entry = by_id[cid]["latest_training_observation"]
            observed = tick.load(closure.artifact(ROOT, entry))
            assert observed["token"] == token and observed["native_progress_confirmed"] is True
            assert observed["pid"] == pid and observed["source_marker"] == commit
            fresh(observed, "observed_local")
            observations[cid] = dict(evidence=entry, phase=observed["after"]["latest"], pid=pid)
            proof_ref = ref("pipeline/checks/gpu_occupation_" + name + "_release_" + token + ".json")
            proof = tick.load(closure.artifact(ROOT, proof_ref))
            fresh(proof, "observed_at")
            candidate = next(row for row in work["candidates"] if row.get("stage") == "blocked"
                and row.get("blocker_kind") == "exclusive_gpu_slot" and row.get("required_server") == name)
            snapshot_server = next(row for row in context["queue"]["servers"] if row["name"] == name)
            state = gpu_occupation.validated_state(ROOT, context, candidate, snapshot_server, queue)
            assert state["busy_gpu_ids"] == state["reserved_gpu_ids"] == [0] and not state["idle_gpu_ids"]
            occupations[name] = proof_ref
        for cid in ("AdaRES-Y94", "AdaRES-Y95"):
            candidate = by_id[cid]
            assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
            assert not candidate.get("native_smoke_result") and not candidate.get("admission_receipt")
            assert not (ROOT / ("pipeline/admissions/" + cid + ".json")).exists()
        y95 = by_id["AdaRES-Y95"]
        validate_cpu(tick.load(closure.artifact(ROOT, y95["CPU_result"])))
        version = tick.load(closure.artifact(ROOT, y95["source_manifest"]))
        validate_binding(version, y95, Y93)
        assert version["CPU_result"] == y95["CPU_result"]
        preparation = tick.load(closure.artifact(ROOT, y95["native_preparation"]))
        assert preparation["guard_tests_passed"] == 10 and preparation["runtime_guard_tests_passed"] == 8
        assert preparation["same_callbacks_actually_executed_on_CPU"] is True and preparation["native_executed"] is False
        for key, count in (("guard_tests", 10), ("runtime_guard_tests", 8)):
            raw = closure.artifact(ROOT, preparation[key]).read_bytes()
            assert ("Ran " + str(count) + " tests").encode() in raw and b"\nOK" in raw and b"skipped" not in raw
        for name, key in (("server-1", "native_smoke_plan_server1"), ("server-2", "native_smoke_plan_server2_gpu0")):
            plan = tick.load(closure.artifact(ROOT, y95[key]))
            source = closure.artifact(ROOT, plan["program"]).read_text(encoding="utf-8")
            frozen = ast.literal_eval(ast.parse(source).body[0].value)
            assert validate_frozen(y95, plan, frozen) == 7
            runtime(name, y95, frozen, servers[name])
        terminal = tick.load(closure.artifact(ROOT, by_id["AdaRES-Y93"]["terminal_handoff_plan"]))
        assert terminal["CPU_guard_tests_passed"] == 29 and terminal["payload"]["allowed_markers"] == [Y93]
        assert terminal["checkpoint_loaded"] is False and terminal["archived"] is False
        assert terminal["training_stopped"] is False and terminal["post_training_freeze"] == by_id["AdaRES-Y93"]["post_training_freeze"]
        for dependency in terminal["dependencies"]:
            closure.artifact(ROOT, dependency)
        location = tick.load(closure.artifact(ROOT, terminal["declared_location"]))
        assert location["checkpoint_directory"] == "20261108_082521" and location["checkpoint_loaded"] is False
        closeout = ROOT / ("pipeline/checks/y95_preparation_closeout_" + token + ".json")
        assert not closeout.exists()
        tick.save(closeout, dict(
            complete=True, token=token, recorded_at=datetime.now().astimezone().isoformat(),
            observations=observations, actual_occupation=occupations, Y95_CPU_result=y95["CPU_result"],
            Y95_retained_version=y95["source_manifest"], Y95_native_preparation=y95["native_preparation"],
            Y93_terminal_plan=by_id["AdaRES-Y93"]["terminal_handoff_plan"],
            earlier_Y94_binding=by_id["AdaRES-Y94"]["additive_version_binding"],
            CPU_guards=10, runtime_guards=8, terminal_guards=29,
            original_Y92_Y93_earlier_Y94_preserved=True, original_full_frozen_FIRST_and_cost_preserved=True,
            original_GPU1_and_Y57_pause_preserved=True, new_scientific_candidates=1,
            actual_new_CPU_candidate_runs=1, actual_GPU_native_tasks=0, new_admissions=0, new_training_launches=0,
            exact_original_archives=0, earlier_CPU_scientific_replays=0,
            original_training_checkpoints_loaded=False, efficacy_measured=False,
            production_HEAD_index_queue_preserved=True, evidence=[ref(__file__), y95["CPU_result"],
                y95["CPU_program"], y95["source_manifest"], y95["native_preparation"],
                by_id["AdaRES-Y93"]["terminal_handoff_plan"]] + [row["evidence"] for row in observations.values()]
                + list(occupations.values())))
        y95["evidence"].append(ref(closeout))
        # The canonical collector's last actual resource_check remains the tick attempt.
        assert work["tick_attempt"]["kind"] == "resource_check"
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, memory_lines = memory(token, view, by_id, observations, occupations, closeout)
        decision = monitor.log_decision(str(ROOT), "next_batch",
            "两授权GPU0仍被原Y92/Y93占用；补充独立待检查候选并保护Y93精确终态。",
            "保留Y94优先；新增Y95全查询均值旁路独立比较，CPU证据及10+8守卫通过后留版；等匹配自然fixed30完整归档和实时GPU0空槽才做原生任务。",
            "原始YAML/fullmodel/发布骨干/TRAIN真实加载、float32/64标量与同焦点改同伴/七阶段W2及输入梯度/因果旁路/两合成更新通过；不是GPU准入或训练收益。",
            closeout.relative_to(ROOT).as_posix())
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        phases = []
        for cid, entry in observations.items():
            phase = entry["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30，步"
                + str(phase["step"]) + "/" + str(phase["count"]))
        tick.save(notes, dict(actions=[
            "仅处理AdaRES_project；本轮所属会话/lease、Disabled后台调度、正式skill/experiment_first、AGENTS/goal/SERVERS/暂停资源记录已核验。begin已先查询进展并尝试已有合格派发，ready waiting为0。",
            "运行：" + "；".join(phases) + "。原PID、source、YAML、receipt、私有runtime、350W/280W及原生步数递增通过，canonical物理占用确认两GPU0 busy+reserved。没有使用GPU1或干扰外部任务。",
            "完成：Y93原生保存声明只读核实实际目录20261108_082521，新建精确终态工具及29项生命周期守卫通过；绑定原PID47380/log/source/YAML/receipt/freeze，不按启动时间猜目录，未加载checkpoint、停止训练或执行归档。旧hash-bound工具保持。",
            "新准备：独立Y95在原self/language和中加入全128查询mean投影旁路，原本地self注意力/SCG/gates保持。首次实际原YAML/完整CPU模型/发布骨干/TRAIN162738点8表达、float32/64独立标量/置换/复制/改同伴保持焦点/零均值/七阶段W2及输入梯度/干预和精确旁路、两合成更新通过；CPU不是稀疏GPU原生任务或收益证据。",
            "Y95只保留七个实际测试新增文件，私有index分支直接基于原生已测Y93；生产HEAD/index/无关worktree/队列和较早Y94保持。10项真实证据守卫与8项runtime/goal/source/owner负例守卫全部通过，无跳过，保存两授权GPU0原生计划及精确runner。GPU任务未执行、未准入、未同步或派发。",
            "本轮新增候选1、CPU候选实测1、GPU原生任务0、准入0、启动0、归档0。原9508表达/141场景完整FIRST/原同设备cost/重复对照/scene bootstrap/有害子组/data/seed1999/fixed30/gates/现有功耗/调度和专用恢复均保持，未声称收益或达标。",
            "记忆已先逐字节备份再压缩为七节短记录，完整历史与负证据沿snapshot链保留。收尾连续刷新实际训练及canonical物理占用并保存真实collector resource_check，随后正式report/finish；没有预先声称postflight通过。"
        ], decisions=[
            "Y94先于Y95；真实原生两步、原full freeze/admission、安全同步和递增counter仍是训练前必需步骤。当前两卡均busy，CPU不代替准入，不挤占现有或外部任务。",
            "未来Y92仅用finalize_y92_fixed30_y93.py和原20261108_090202；Y93用本轮finalize_y93_fixed30.py/29守卫和原20261108_082521。自然fixed30完整checkpoint核验后方可归档；任何新增HEAD/owner需新的已测绑定。已归档Y91和更早实验不重放。",
            "server2 GPU1全部使用与原GPU1后评/cost继续用户暂停，不迁移替代；Y57单独暂停，各自恢复需用户明确授权。禁止server6，不改变科学goal字节、原receipt、data/runtime/mappings、功率或调度；无需新增用户处理。"
        ], evidence=[ref(closeout), backup, ref(decision)]))
        assert all(ref(path) == value for path, value in before.items())
        return closeout, notes, memory_lines
    finally:
        writer.close()


def finish(token):
    lease(token, 240)
    for cid in ("AdaRES-Y92", "AdaRES-Y93"):
        command(["scripts/y93_deployment.py", "observe", "--id", cid, "--token", token, "--label", "release"])
    for name in ("server-1", "server-2"):
        command([SKILL + "/gpu_occupation.py", "AdaRES_project", "--token", token,
                 "--server", name, "--label", "release"])
    closeout, notes, memory_lines = record(token)
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
    print(json.dumps(dict(closeout=ref(closeout), observations=tick.load(closeout)["observations"],
        receipt_status=receipt["status"], postflight=receipt["postflight"], context_committed=True,
        lease_release=receipt["lease_release"], memory_lines=memory_lines,
        archive_pending_files=0, notes=ref(notes)), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    finish(parser.parse_args().token)
