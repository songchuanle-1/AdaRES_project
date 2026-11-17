"""Close CPU preparation with current physical occupation and protected runs."""
import argparse
import ast
import base64
from datetime import datetime
import json
import os
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y93_deployment import Y92, Y93, contract, git
from prepare_y98_appearance_mask_skip import files
from y98_contract import validate_cpu
from run_y98_native import runtime, validate_binding
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
    lines = old[:old.index("## Last Tick")]
    for index, line in enumerate(lines):
        if line.startswith("- Current token "):
            lines[index] = "- Current token " + token + "; snapshot " + view["generated_at"] + "; deadline " + view["deadline_at"] + "; never extend."
        elif line.startswith("- Exact previous seven-section memory:"):
            lines[index] = "- Exact previous seven-section memory: " + evidence(ref(backup)) + "; complete prior snapshot-chain history/bans/negative evidence/dedicated recovery requirements retained byte-for-byte."
        elif line.startswith("- Current AdaRES-Y92 ") or line.startswith("- Current AdaRES-Y93 "):
            cid = line.split()[2]
            entry = observations[cid]
            phase = entry["phase"]
            lines[index] = "- Current " + cid + " " + phase["phase"] + str(phase["epoch"]) + "/30 step" + str(phase["step"]) + "/" + str(phase["count"]) + "; original native progress/source/YAML/runtime/power verified: " + evidence(entry["evidence"])
        elif line.startswith("- Fresh canonical occupation:"):
            lines[index] = "- Fresh canonical occupation: " + "; ".join(evidence(entry) for entry in occupations.values()) + "; both permitted GPU0 busy/reserved, age<300s required. Keep collector's actual tick_attempt and all candidate evidence."
        elif line.startswith("- Next tick first observeY92/Y93"):
            lines[index] = "- Next tick first observeY92/Y93 and dispatch ready work; Y94 thenY95 thenY96 thenY98 remain CPU-prepared, GPU-unexecuted/unadmitted. Matching owner natural30/full checkpoint/archive plus fresh idle GPU0 required. Y97 original+revision1 failures stay immutable numerical holds. No timed CPU/version/plan/terminal replay; changed HEAD/owner needs NEW tested bindings."
    where = lines.index("## Active Hypotheses") + 1
    result = tick.load(closure.artifact(ROOT, new["CPU_result"]))
    lines[where:where] = [
        "- Y98 independent W2 direct full-existing-appearance static-mask residual: mask_static=x_mask(x)+raw_existing_2D@zeroW1024x256 once before original batching/seven dots; +262144/total164891781/no RNG. Complete original x_mask and input_proj_2d/visual fusion, all1024 existing2D channels/all sparse features/points, query norms/scores/shared confidence/refinement/sampling/SCG/gate algorithm/targets-loss-selector/releasedbackbone/seed1999/fixed30/native YAML unchanged. No new data or feature/weight/threshold/validation scan.",
        "- Y98 is NOT existing Y71 initial 3D/2D product, Y72 recurrent refined-visual mask update, Y81 query-attended mask feature, Y68 language modulation, Y85 mask norm, Y88/Y92 quadratic readouts or Y96 seed skip. No combination/reopening/missing-information/defect/novelty/benefit claim. The duplicate initial-product idea was recognized as Y71 BEFORE any new code/ID/test; Y97 failures and guards untouched.",
        "- Y98 FIRST actual original CPU YAML/fullmodel/releasedbackbone/TRAIN162738points8expressions passed: independent float32/64 scalar/no-RNG/row-channel permutation/zero/sign/row-local/both-input gradients, exact seven-stage W2/RNG/weight and close input-gradient anchors, initial/final mask intervention/exact initial-score preservation/bypass, same full1024 raw2D/base/batching/static seven consumers and two finite synthetic updates with387 gradient tensors. Learned mask-feature delta " + str(result["CPU_appearance_mask_paths"][1]["mask_feature_delta_max"]) + ". CPU result " + evidence(new["CPU_result"]) + "; program " + evidence(new["CPU_program"]) + ". NOT sparse GPU/native/admission/efficacy.",
        "- Y98 retained ONLY seven tested A files commit" + new["code_commit"] + "/" + new["retained_branch"] + " directly on native-trained HEADY93; main HEAD/index/queue/earlierY94/Y95/Y96 preserved. Version " + evidence(new["source_manifest"]) + "; native preparation " + evidence(new["native_preparation"]) + ". Ten actual-evidence and8 runtime guards passed FIRST execution/no skip. Same callback body actually CPU-executed; GPU unexecuted.",
        "- Y98 runner=run_y98_native.py ONLY at HEADY93 after matching Y92/server1 orY93/server2 natural30/full exact finite checkpoint/canonical archive plus fresh idle/unreserved authorizedGPU0; earlierY94/Y95/Y96 precedeY98. Further HEAD/owner changes require NEW tested binding. Actual native PASS/full original post-training freeze/receipt/admission/exact source selection/safe sync/increasing native counters remain required; no CPU/version/plan replay.",
    ]
    lines += [
        "## Last Tick",
        "- " + token + ": original Y92/Y93 native progress/source/YAML/receipts/private runtimes/350W-280W verified; no terminal, restart, checkpoint loading, stop or archive. Y94/Y95/Y96 original preparation and Y97 original/revision1 numerical holds preserved without replay.",
        "- New independent Y98 full-existing-2D appearance connection to static masks implemented; FIRST real CPU model/YAML/backbone/TRAIN collation and full component/decoder/consumer/two-update controls passed. Only tested files retained separately;10 actual-evidence plus8 runtime guards passed, current-owner plans prepared for permittedGPU0 only.",
        "- New scientific IDs1/full CPU attempts1(passed)/GPU native0/admissions0/launches0/archives0. Both permittedGPU0 still busy+reserved; Y98 cannot claim native PASS or queue admission. No training-best/FIRST promotion, GPU1 use, secondY97 revision or new user authorization request.",
        "- All scientific goal bytes/old receipts/full original FIRST and original-device cost/data/runtime/mappings/seed1999-fixed30/bans/power/schedule/restore rules remain unchanged. GPU1 all use and original GPU1 frozen review stay paused; Y57 independently paused; no resource substitution or foreign-job interference.",
        "- Closeout " + evidence(ref(output)) + "; exact memory backup, sequential current native observations/canonical occupation, keep collector's actual resource_check, official Chinese report/finish. No preclaim; verify FINALIZED/PASS/context_committed/released and absent lock/pending/archive journals read-only after finish.",
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
        observations, occupations = {}, {}
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
            occupations[name] = proof_ref
            terminal = tick.load(closure.artifact(ROOT, candidates[cid]["terminal_handoff_plan"]))
            assert terminal["checkpoint_loaded"] is False and terminal["archived"] is False
            assert terminal["post_training_freeze"] == candidates[cid]["post_training_freeze"]
        earlier = {}
        for cid in ("AdaRES-Y94", "AdaRES-Y95", "AdaRES-Y96"):
            row = candidates[cid]
            assert row["stage"] == "blocked" and row["blocker_kind"] == "exclusive_gpu_slot"
            assert not row.get("native_smoke_result") and not row.get("admission_receipt")
            for key in ("CPU_result", "source_manifest", "native_preparation"):
                closure.artifact(ROOT, row[key])
            earlier[cid] = row["source_manifest"]
        held = candidates["AdaRES-Y97"]
        assert held["stage"] == "blocked" and held["blocker_kind"] == "numerical_control_failure"
        assert held["failure_scope"] == "implementation" and len(held["revision_history"]) == 1
        assert held["revision_history"][0]["result"] == "failed"
        closure.validate_revisions(ROOT, held)
        for key in ("CPU_failure", "CPU_program", "component_diagnosis", "revision1_result", "revision1_CPU_program"):
            closure.artifact(ROOT, held[key])
        registration = tick.load(closure.artifact(ROOT, held["revision1_registration"]))
        for entry in registration["original_files"]:
            closure.artifact(ROOT, entry)
        row = candidates["AdaRES-Y98"]
        assert row["stage"] == "blocked" and row["blocker_kind"] == "exclusive_gpu_slot"
        assert not row.get("native_smoke_result") and not row.get("admission_receipt")
        assert not (ROOT / "pipeline/admissions/AdaRES-Y98.json").exists()
        cpu = tick.load(closure.artifact(ROOT, row["CPU_result"]))
        validate_cpu(cpu)
        version = tick.load(closure.artifact(ROOT, row["source_manifest"]))
        validate_binding(version, row, Y93)
        preparation = tick.load(closure.artifact(ROOT, row["native_preparation"]))
        assert preparation["guard_tests_passed"] == 10 and preparation["runtime_guard_tests_passed"] == 8
        assert preparation["same_callbacks_actually_executed_on_CPU"] is True
        assert preparation["native_observer_callbacks_executed_on_GPU"] is False
        for key in ("guard_tests", "runtime_guard_tests"):
            raw = closure.artifact(ROOT, preparation[key]).read_bytes()
            assert b"OK" in raw and b"skipped" not in raw
        payload = tick.load(closure.artifact(ROOT, version["source_snapshot"]))["CPU_tested_payload"]
        program = closure.artifact(ROOT, row["CPU_program"]).read_text(encoding="utf-8")
        assert ast.literal_eval(ast.parse(program).body[0].value) == payload
        for path in files():
            value = payload["variant"]["config_data"] if path.suffix == ".yaml" else payload["variant"]["python_files"][path.name]
            assert path.read_bytes() == base64.b64decode(value["base64"]) and ref(path)["sha256"] == value["sha256"]
            assert path.read_bytes() == git("show", row["code_commit"] + ":" + path.relative_to(ROOT).as_posix())
        assert git("rev-parse", row["code_commit"] + "^").decode().strip() == Y93
        core = closure.artifact(ROOT, row["observer_core"]).read_text(encoding="utf-8")
        body = closure.artifact(ROOT, row["prepared_native_body"]).read_text(encoding="utf-8")
        assert program.count(core) == body.count(core) == 1
        plans = {}
        for name, key in (("server-1", "native_smoke_plan_server1"), ("server-2", "native_smoke_plan_server2_gpu0")):
            plan = tick.load(closure.artifact(ROOT, row[key]))
            source = closure.artifact(ROOT, plan["program"]).read_text(encoding="utf-8")
            frozen = ast.literal_eval(ast.parse(source).body[0].value)
            assert validate_frozen(row, plan, frozen) == 7
            assert runtime(name, row, frozen, servers[name]) == (350.0 if name == "server-1" else 280.0)
            assert plan["execution_helper"] == ref("scripts/run_y98_native.py") and plan["additive_parent"] == Y93
            plans[name] = row[key]
        output = ROOT / ("pipeline/checks/y98_preparation_closeout_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
            observations=observations, occupations=occupations, earlier_preparations=earlier,
            new_candidate="AdaRES-Y98", CPU_result=row["CPU_result"], CPU_version=row["source_manifest"],
            native_preparation=row["native_preparation"], current_owner_native_plans=plans,
            original_Y97_numerical_failures_unchanged=True, original_Y92_Y93_preserved=True,
            new_scientific_candidates=1, CPU_candidate_attempts=1, CPU_candidate_attempt_result="passed",
            actual_evidence_guards=10, runtime_guards=8, native_GPU_tasks=0, new_admissions=0,
            new_training_launches=0, original_archives=0, original_training_checkpoints_loaded=False,
            original_full_frozen_FIRST_and_cost_preserved=True, original_GPU1_and_Y57_pause_preserved=True,
            efficacy_measured=False, production_HEAD_index_queue_preserved=True,
            evidence=[ref(__file__), row["CPU_result"], row["CPU_program"], row["source_manifest"],
                row["native_preparation"], held["CPU_failure"], held["revision1_result"]]
                + [entry["evidence"] for entry in observations.values()] + list(occupations.values()) + list(plans.values())))
        row["evidence"].append(ref(output))
        assert work["tick_attempt"]["kind"] == "resource_check"
        tick.save(ROOT / "pipeline/research_work.json", work)
        backup, lines = memory(token, view, row, observations, occupations, output)
        decision = monitor.log_decision(str(ROOT), "next_batch",
            "授权GPU0继续训练、已有候选未能做原生检查时，如何准备独立结构而不重放旧实验？",
            "保留Y94/Y95/Y96优先顺序和Y97数值挂起，准备独立Y98全现有2D外观到静态mask残差；只保存首次CPU通过的七文件及GPU0检查计划。",
            "原生YAML/完整模型/发布骨干/TRAIN加载、float32/64标量与行局部控制、W2锚点/旁路、两步有限合成decoder更新和同一消费者回调通过；10+8守卫通过。GPU0仍busy+reserved，尚无真实稀疏GPU task，不准入不入队。",
            output.relative_to(ROOT).as_posix())
        phases = []
        for cid, entry in observations.items():
            phase = entry["phase"]
            phases.append(cid + " " + phase["phase"] + str(phase["epoch"]) + "/30，步"
                + str(phase["step"]) + "/" + str(phase["count"]))
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "仅处理AdaRES_project；owner实际会话、Disabled后台调度、正式skill/experiment_first、AGENTS/goal/SERVERS及最新GPU1暂停记录核验后begin，先查进展并尝试已有合格派发；ready waiting为0，无新终态。",
            "运行：" + "；".join(phases) + "；原PID/source/YAML/receipt/私有runtime、350W/280W及原生步数递增核实。canonical物理占用确认两个授权GPU0均busy+reserved；GPU1和外部进程未干扰。",
            "新准备：Y98独立全现有1024通道2D外观到静态mask特征的零初始化残差，一次接入原batching供七次mask读取；保留完整原x_mask和2D视觉融合、全部输入/SCG/采样/细粗融合/共享头/硬门算法/目标损失选择器/seed1999固定30轮及原YAML。不是已有Y71视觉融合乘积或Y72/Y81循环mask更新，不重开或组合旧路线。",
            "首次真实CPU检查通过：原生YAML、完整模型、发布骨干和TRAIN162738点8表达加载；float32/64独立标量/无RNG/行通道置换/零输入/符号/行局部/双输入梯度，七阶段W2精确锚点/输入梯度、初始mask干预且初始score不变/精确旁路，完整原2D/base/batching/静态七次复用和两步有限合成decoder更新。CPU不是GPU稀疏task或训练收益。",
            "仅七个已测A文件独立留版，生产HEAD和主index保持；10项实际证据检查和8项运行边界检查首次执行通过、无跳过。为当前Y92/server1和Y93/server2分别绑定GPU0原生检查计划，仅允许原数据、私有环境、既有路径映射和350W/280W。没有实际GPU原生、准入、同步、入队或启动。",
            "Y94/Y95/Y96原CPU版本/原生计划保留不重放；Y97原始与修订1失败、完整原守卫和文件不变，不新增修订或重命名。两项当前训练终态计划只读核验，未读checkpoint、停止或归档，早期归档不重放。",
            "本轮新科学ID1、完整CPU尝试1通过、GPU原生0、准入0、启动0、归档0；精确备份七节记忆、刷新训练和canonical占用、保留collector实际resource_check，再正式report/finish；不预先宣称收尾结果。"
        ], decisions=[
            "Y98当前仅完成CPU准备，不是原生检查PASS；保持blocked/exclusive_gpu_slot，不伪造waiting。先保留Y94、Y95、Y96顺序，再考虑Y98；匹配owner自然30轮、完整有限checkpoint精确终态核验归档及实时空闲GPU0后才做真实稀疏原生两步。",
            "不得重放已完成CPU/version/plan或已归档终态检查；若生产HEAD或资源owner改变，须新测试绑定再执行候选。实际native通过后仍须完整原FIRST/同设备cost冻结、准入收据、精确源选择/安全同步和原生递增步数验证。无需新增用户授权。",
            "科学goal字节、原签名收据、全部数据/私有环境/mappings/seed1999固定30轮/禁用/恢复/功耗和调度要求不变。GPU1全部使用与原GPU1冻结后评成本继续暂停，Y57另行暂停，不迁移替代，不用server6或跨项目。"
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
