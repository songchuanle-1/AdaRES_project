"""Bind this tick's production CPU controls and live, bounded GPU blocker."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
prefix = "pipeline/analyses/"


def evidence(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def result(stem):
    path = prefix + stem + "_" + args.token + ".json"
    value = json.loads((ROOT / path).read_text())
    assert value["complete"]
    return evidence(path), value


def save(path, value):
    with (ROOT / path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return evidence(path)


upload_ref, upload = result("y46_production_upload_v2_131e")
cpu_ref, cpu = result("y46_production_cpu_131e")
live_ref, live = result("y48_live_resources_131e")
assert cpu["CPU_hook_cases"] == cpu["CPU_captured_expressions"] == 32
assert cpu["physical_DEC_parent_initialization_exact"] and cpu["RNG_preserved"]
assert cpu["exact_production_vs_uncached_all_expressions"]
assert cpu["new_tensors"] == 8 and cpu["added_parameters"] == 328961
assert live["native_steps_increased"] and live["pid"] == 8877
assert all(row["pids"] for row in live["second"]["GPUs"])
old = "601bcda714184133920945f8f067b382"
old_y48 = json.loads((ROOT / (prefix + "y48_local_bundle_601b_" + old + ".json")).read_text())
expected = {row["path"].removeprefix("AdaRES/"): row["sha256"] for row in old_y48["files"]}
assert live["source_hashes"] == expected
assert cpu["production_source_hashes"] == {
    row["path"].removeprefix("AdaRES/"): row["sha256"]
    for row in upload["files"] if row["path"].endswith(".py")}
native_program = "scripts/probe_y46_production_native_131e.py"
compile((ROOT / native_program).read_text(encoding="utf-8"), native_program, "exec")
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    failure_refs = [
        evidence(prefix + "upload_y46_failed_guard_131e_" + args.token + ".py"),
        evidence(prefix + "y46_production_upload_131e_" + args.token + ".log")]
    repair_ref = save(prefix + "y46_upload_guard_repair_131e_" + args.token + ".json", dict(
        complete=True, id="AdaRES-Y46", failure_evidence=failure_refs,
        result="Original guard failed before any upload because /proc initial environ lacks CUDA_VISIBLE_DEVICES.",
        repair="Exact tools/train.py config and --gpu_ids=1 plus nvidia-smi PID/card mapping replace the invalid initial-environ assumption.",
        model_or_protocol_changed=False, pre_upload_failure=True,
        retained_failed_source_sha256=failure_refs[0]["sha256"],
        repaired_upload=upload_ref, protected_files=upload["protected_files"],
        protected_sources_unchanged=upload["protected_sources_unchanged"]))
    now = datetime.now().astimezone()
    freeze_ref = save(prefix + "y46_next_production_native_131e_" + args.token + ".json", dict(
        complete=True, id="AdaRES-Y46", preregistered_at=now.isoformat(),
        implementation_revision=1, production_files=upload["files"],
        production_source_receipt=upload_ref, production_CPU_controls=cpu_ref,
        next_program=evidence(native_program),
        next_program_compiled=True, next_program_executed=False,
        test="Actual independent production released-SPFormer initialization; frozen TRAIN indices0,44,60,101 in two native batches; tools.train2 optimizer updates on all8 new tensors; exact saved model/AdamW/PolyLR restore including actual HF metadata tensors; hook registration after restore.",
        subsequent_required_tests=[
            "Same frozen TRAIN8 native matched disabled/wrong/cached-repeat controls and all8 native gradients.",
            "Paired serial inference batch1, active128 queries/single selection and original parent comparison.",
            "Parent/candidate native stress on frozen indices728/6780; report warm paired time and peak memory.",
            "Six-gate substantive review, fixed post-training plan and tested exact source commit before any receipt or queue."],
        allocation="Only a live-idle server2 GPU0 or GPU1 with no compute process and an acquired exclusive diagnostic lease; preserve every foreign/current job.",
        runner="scripts/run_adares_authorized_probe.py y46_production_native_131e scripts/probe_y46_production_native_131e.py --token <fresh-owned-token> --seconds 900 --gpu --gpu-id <live-idle-authorized-card>",
        limitations="CPU preparation only. Production native backward, optimizer updates and GPU costs remain UNRUN, not PASS. Original3.49-18.66s and cached0.627-8.277s vs parent0.213-0.297s cost risks preserved; no second revision or validation selection."))
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    y46 = next(candidate for candidate in work["candidates"] if candidate["id"] == "AdaRES-Y46")
    y48 = next(candidate for candidate in work["candidates"] if candidate["id"] == "AdaRES-Y48")
    assert y46["stage"] == "implementation" and y48["stage"] == "training"
    assert len(y46["revision_history"]) == 1 and y46["revision_history"][0]["result"] == "passed"
    y46.update(
        stage="blocked", scientific_status="production_CPU_controls_passed_GPU_validation_blocked",
        blocker_kind="live_external_gpu_occupation", required_server="server-2",
        blocker="Both authorized cards have live compute PIDs: GPU0 foreign19723; GPU1 SAME verified Y48 PID8877. Production native backward/optimizer/restore/resources require an idle exclusively leased GPU. Necessary source/CPU preparation for that next test is complete.",
        retry_after=(now + timedelta(hours=1)).isoformat(),
        unblock_condition="An authorized server2 GPU0/1 becomes live-idle with no compute PID; acquire its exclusive diagnostic lease. A timer/queue capacity alone is not proof of a free GPU.",
        next_action="Run SAME hash-frozen production native program on first genuinely idle authorized card, then matched TRAIN8/inference/stress and six-gate review. Do not repeat unchanged CPU tests or admit from prototype.",
        attempted_alternatives=[
            "Exactly seven additive production files uploaded with21 existing source/config bytes protected.",
            "Actual independent CPU import/DEC construction; all inherited state and RNG exactly preserved, eight registered new tensors.",
            "Actual production hook32 CE controls: final-only/disable/repeat/wrong correspondence.",
            "All32 captured native TRAIN expressions: production rows/weights/groups/scores exactly equal uncached implementation; finite nonzero CPU square gradients.",
            "Frozen native optimizer/restore program compiled, not run on occupied GPUs."],
        pending_gates=["production_native_backward", "production_tools_train_and_restore", "resources", "final_six_gate_review"],
        diagnostic_gates={
            "diagnosis": "prior prediction-blind information witnesses retained; no efficacy or universal blindness claim",
            "controls": "actual independent production CPU controls passed; full native controls still pending",
            "independence": "physical six sources/config/import/parent-stripped AST/constructor checked; final native package test pending",
            "resources": "added328961 parameters verified, but runtime/memory admission unresolved"},
        failure_scope="implementation")
    refs = [upload_ref, cpu_ref, live_ref, repair_ref, freeze_ref,
            evidence("scripts/upload_y46_production_131e.py"),
            evidence("scripts/probe_y46_production_cpu_131e.py"),
            evidence(native_program), *failure_refs]
    y46["evidence"].extend(ref for ref in refs if ref not in y46["evidence"])
    if live_ref not in y48["evidence"]:
        y48["evidence"].append(live_ref)
    work["design_cycle"] = dict(
        stop_reason="external_resource_blocked", attempted_ids=["AdaRES-Y46"],
        evidence=[live_ref, cpu_ref, upload_ref, freeze_ref],
        retry_after=(now + timedelta(hours=1)).isoformat(),
        resume_condition=y46["unblock_condition"] + " Resume frozen native test before any admission.",
        rationale="No free physical GPU despite one logical queue vacancy. This is a bounded resource stop, not hypothesis exhaustion; reserve budget is a maximum and does not justify placeholder variants.")
    path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    question = "Y46独立生产CPU控制已通过，但授权两卡都有真实进程，能否直接准入？"
    decision = "不准入、不排队、不重启。原稳定Y46保留修订1，等待真实独占空卡补生产原生、优化器恢复与资源门槛；保护Y48及外部任务。"
    rationale = "生产32例钩子与32原生捕获表达式已实测；新增328961参数，父初始化/RNG精确一致。GPU0 PID19723、GPU1 Y48 PID8877；原运行时间风险未解决。"
    logged = subprocess.run(
        ["C:/Program Files (x86)/Git/bin/bash.exe",
         "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/log_decision.sh",
         "AdaRES_project", "other", question, decision, rationale,
         "; ".join(ref["path"] for ref in (cpu_ref, upload_ref, live_ref, freeze_ref))],
        capture_output=True, encoding="utf-8", timeout=30)
    assert logged.returncode == 0, logged.stderr
    actions = [
        "本轮设计快照2026-10-15 11:53:18+08，截止12:53:18+08；一次begin，未重复preflight。0-3段是该快照，以下才是后续实测；本报告生成时finish尚未执行，最终执行状态以命令与收据为准。",
        "11:58仅新增上传原稳定Y46七个独立生产文件；逐字节保护W2/Y45/正在训练Y48共21个源文件及配置，当前提交标记d18c228不变；无提交、准入、排队或新训练。",
        "11:58:19-12:00:48本地观测窗口：实际独立包CPU导入/父AST/构造器参数与RNG通过；新增328961参数、8张量。32例生产钩子仅最终前景改变，六辅助/禁用父路径/重复精确一致，错配差异最小0.029464993；代理CE最小梯度0.0001444601。不是原生训练证据。",
        "全部8场景32条原生TRAIN捕获表达式，生产组件行/质量权重/分组/输出与未缓存原实现精确一致；生产读数器CPU平方损失最小梯度0.0025017884。独立生产全模型原生反传、真实tools.train、恢复与资源测试仍未知/未执行，不转为PASS。",
        "保留上传守卫初次KeyError的原源与日志：进程初始环境没有CUDA_VISIBLE_DEVICES；仅改为精确argv --gpu_ids=1和显卡驱动PID对应校验。失败发生在上传前，模型/门槛/协议未改变；修复后七文件上传成功。",
        "12:02:47-12:02:53本地观测：Y48原PID8877、GPU1、配置/七文件/提交标记吻合，原生第1/30轮2783→2792/4686；GPU0外部PID19723、89%、21946/24576MiB，GPU1 Y48、80%、17410/24576MiB。两卡均占用，无GPU诊断/抢占/盲重启。",
        "原稳定Y46的下一生产原生两步/完整模型AdamW/PolyLR恢复程序已冻结并通过语法编译，未执行；后续仍须TRAIN8匹配控制、串行推理与最大双场景压力/成本审查。旧缓存修订1和原始3.49-18.66s、缓存0.627-8.277s对父0.213-0.297s的成本风险全部保留。",
        "科学状态：Y48=training_not_efficacy，Y46=production_CPU_controls_passed_GPU_validation_blocked；Y45固定30轮不确定结论复用既有冻结审查，不重测、不晋升。实际新候选0、准入0、新训练0；Y46推进及Y48进度证据已持久化。",
        "本轮收据尚未生成，用量窗口/调用数/输入缓存/输出及收据产出计数暂未知；finish后只读收据核实，不回写已释放项目。用量仅begin至收据本地可见窗口，缓存属于输入，推理属于输出，非账单；初始读取/最终回复/迟到记录不计入。",
    ]
    decisions = [
        "按真实双卡占用使用external_resource_blocked收尾，不等待截止、不降低门槛、不宣称路线穷尽。Y46设置1小时有界复查；真实资源或项目证据变化可提前唤醒，仍须现场空闲和独占租约。",
        "保留同一Y46 ID和已通过修订1；本轮没有第二修订、参数扫描或占位变体。储备1/2、准备预算1不是必须造满；训练准入预算1未使用。",
        "Y48维持原released-SPFormer/seed1999/固定30轮/单卡训练，终态后执行已冻结全9508/141两遍各方法FIRST主结果、W2/B1、场景区间/子组伤害/成本；不做验证调参。",
        "仅AdaRES_project/server2 GPU0、1；后台调度保持Disabled，未接触其他项目、server6、调度或新任务。下一实际自动化时间未知。",
        "命令finish必须明确PASS/COMPLETE且匹配租约消失后才称本轮COMPLETED；否则保留以上证据并按契约abort。"]
    notes = ROOT / ("pipeline/headless/.notes_" + args.token + ".json")
    with notes.open("x", encoding="utf-8") as stream:
        json.dump(dict(actions=actions, decisions=decisions), stream, indent=2, ensure_ascii=False)
    print(json.dumps(dict(complete=True, id="AdaRES-Y46", stage="blocked",
                         blocker_kind=y46["blocker_kind"], retry_after=y46["retry_after"],
                         next_native_test=freeze_ref, report_notes=str(notes),
                         decision_log=logged.stdout.strip()), ensure_ascii=False, indent=2))
finally:
    writer.close()
