"""Record a due candidate's real occupied-GPU recheck without rerunning controls."""

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
parser.add_argument("--candidate-id", required=True)
parser.add_argument("--resource", required=True)
parser.add_argument("--next-test", required=True)
parser.add_argument("--observed-start", required=True)
parser.add_argument("--observed-end", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
now = datetime.now().astimezone()
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
start, end = map(datetime.fromisoformat, (args.observed_start, args.observed_end))
assert start.tzinfo is not None and end.tzinfo is not None
assert datetime.fromisoformat(owner["started_at"]) <= start <= end <= now
assert (now - end).total_seconds() < 600, "Resource observation too old"


def artifact(path):
    resolved = (ROOT / path).resolve()
    assert ROOT in resolved.parents and resolved.is_file()
    return resolved, dict(path=resolved.relative_to(ROOT).as_posix(),
                          sha256=hashlib.sha256(resolved.read_bytes()).hexdigest())


resource_path, resource_ref = artifact(args.resource)
assert args.token in resource_path.name
observed = json.loads(resource_path.read_text(encoding="utf-8"))
assert observed["complete"] and observed["native_steps_increased"]
assert observed["free_authorized_gpus"] == []
assert all(row["pids"] for row in observed["second"]["GPUs"])
next_path, next_ref = artifact(args.next_test)
frozen = json.loads(next_path.read_text(encoding="utf-8"))
assert frozen["complete"] and frozen["id"] == args.candidate_id
program = frozen["next_program"] if "next_program" in frozen else frozen["program"]
program_path, actual_program_ref = artifact(program["path"])
assert actual_program_ref == program
compile(program_path.read_text(encoding="utf-8"), str(program_path), "exec")
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidate = next(row for row in work["candidates"] if row["id"] == args.candidate_id)
    assert candidate["stage"] == "blocked"
    assert datetime.fromisoformat(candidate["retry_after"]) <= now
    assert candidate["blocker_kind"] == "live_external_gpu_occupation"
    assert next_ref in candidate["evidence"] and actual_program_ref in candidate["evidence"]
    queue_path = ROOT / "pipeline/queue.json"
    queue_hash = hashlib.sha256(queue_path.read_bytes()).hexdigest()
    task = next(row for row in monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"]
                if row["id"] == observed["id"])
    assert task["status"] == "running" and task["launch_verified"] and int(task["gpu"]) == 1
    assert task["pid"] == observed["original_pid"]
    assert task["pid"] in observed["second"]["GPUs"][1]["pids"]
    assert task["pid"] not in observed["second"]["GPUs"][0]["pids"]
    for path, digest in observed["source_hashes"].items():
        blob = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", task["code_commit"] + ":AdaRES/" + path])
        assert hashlib.sha256(blob).hexdigest() == digest
    retry = (now + timedelta(hours=1)).isoformat()
    disposition_path = ROOT / ("pipeline/analyses/resource_block_disposition_" + args.token + ".json")
    with disposition_path.open("x", encoding="utf-8") as stream:
        json.dump(dict(
            complete=True, candidate_id=args.candidate_id, resource_evidence=resource_ref,
            observed_start_local=args.observed_start, observed_end_local=args.observed_end,
            frozen_next_test=next_ref, unchanged_program=actual_program_ref,
            native_program_compiled_not_executed=True,
            unchanged_scientific_status=candidate["scientific_status"],
            pending_gates=candidate["pending_gates"], retry_after=retry,
            repeated_CPU_or_native_controls=False, new_candidates=0, new_admissions=0,
            scientific_scope="Fresh due resource recheck only; not a native/efficacy/resource-gate PASS.",
            current_task_code_commit=task["code_commit"],
            current_task_exact_source_hashes_verified=True), stream, indent=2)
    _, disposition_ref = artifact(disposition_path)
    candidate.update(
        blocker="Fresh local " + args.observed_end
        + ": both authorized server2 GPUs still have live compute PIDs; GPU0 foreign19723, GPU1 originalY48 PID8877. SAME frozen production native test needs an idle exclusively leased GPU; prior required CPU/source controls remain complete.",
        retry_after=retry,
        unblock_condition="A server2 GPU0/1 is freshly live-idle with no compute PID and an acquired exclusive diagnostic lease. Preserve foreign/current jobs; a timer, queue vacancy or old snapshot does not grant a GPU.")
    candidate["evidence"].extend([resource_ref, disposition_ref])
    running = next(row for row in work["candidates"] if row["id"] == observed["id"])
    running["evidence"].append(resource_ref)
    work["design_cycle"] = dict(
        stop_reason="external_resource_blocked", attempted_ids=[args.candidate_id],
        evidence=[resource_ref, disposition_ref, next_ref],
        resume_condition=candidate["unblock_condition"] + " Resume existing Y46/Y49 before expanding reserve.",
        retry_after=retry,
        rationale="Due frozen test remains physically blocked; research reserve2/2 and new reserve budget0. No repeated unchanged review, variant filling, fake exhaustion or deadline padding.")
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    assert hashlib.sha256(queue_path.read_bytes()).hexdigest() == queue_hash
    view = json.loads((ROOT / "pipeline/headless/.tick_model_view.json").read_text(encoding="utf-8"))
    second = observed["second"]
    notes = dict(actions=[
        "本轮设计快照" + view["generated_at"] + "，真实截止" + view["deadline_at"]
        + "；一次begin/preflight，0-3段复用脚本快照。报告保存时finish尚未执行，最终执行状态以命令/收据及租约释放核验为准。",
        "紧凑快照显示仅Y46到期、Y49尚未到复查时间，研究储备2/2、准备预算0、准入预算1。只续查原稳定Y46的外部资源阻塞，不新增占位候选或重做已通过的CPU控制。",
        "实际观测窗口" + args.observed_start + "至" + args.observed_end
        + "：原Y48 PID8877/cwd/配置/argvGPU1成立，七文件哈希与原准入代码提交完全一致；原生第"
        + str(second["native_step"][0]) + "/30轮步数"
        + str(observed["first"]["native_step"][2]) + "→" + str(second["native_step"][2])
        + "/4686，训练正常推进，不是重启。",
        "该窗口GPU0外部PID19723，利用率" + str(second["GPUs"][0]["utilization"])
        + "%，显存" + str(second["GPUs"][0]["memory_MiB"]) + "/24576MiB；GPU1原Y48，利用率"
        + str(second["GPUs"][1]["utilization"]) + "%，显存"
        + str(second["GPUs"][1]["memory_MiB"]) + "/24576MiB。实际空闲授权卡0；未分配CUDA诊断上下文、未抢占外部任务或重启原训练。",
        "Y46下一原生生产程序与冻结清单哈希未变，语法编译通过但没有执行。修订1/生产CPU证据沿用，原始与缓存成本风险、生产native loss/backward/实际tools.train/完整恢复/资源/最终审查待办不变；没有把复查当门槛PASS。",
        "科学状态仍为Y48=training_not_efficacy，Y46/Y49=blocked。Y46下次有界复查" + retry
        + "；真实资源/证据变化可提前唤醒，仍须现场空闲与独占租约。Y49原复查时间保持不变。",
        "本轮新候选0、准入0、新训练0；完成1次到期资源复查和冻结证据绑定。没有假设修改、第二实现修订、验证调参、终态重测或其他项目/调度/后台操作。",
        "收据尚未生成，调用数、输入/缓存/输出及收据产出统计暂未知；finish后只读核实，不在释放租约后回写。窗口仅begin至收据的本地可见记录，缓存属于输入、推理属于输出，不重复相加或当账单；早期读取/最终回复/迟到记录不计。"
    ], decisions=[
        "本轮没有新科学决策；按现场双卡占用维持原Y46阻塞，以external_resource_blocked收尾。储备已满，不重复设计、凑变体、降低准入门槛或等待截止；本轮复查不证明路线穷尽。",
        "Y46按同ID、同修订1、同冻结生产原生程序续做；Y49未到期，不重查其CPU/原生证据。出现真实空卡后先续旧候选，不因逻辑队列空位或计时器到期直接准入。",
        "保护原Y48单卡/released-SPFormer/seed1999/固定30轮；原终态冻结评价计划与W2科学比较对象不变。早期验证分数不是最终效用或晋升证据。",
        "仅AdaRES_project及授权server2 GPU0、1，后台调度保持Disabled；没有其他项目/server6/新任务/嵌套执行。下一保存自动化时刻未知。",
        "只有finish明确PASS/COMPLETE并独立核实匹配租约消失才称本轮COMPLETED；否则保存证据并abort，不将科学blocked误写为执行失败或成功门槛。"
    ])
    notes_path = ROOT / ("pipeline/headless/.notes_" + args.token + ".json")
    with notes_path.open("x", encoding="utf-8") as stream:
        json.dump(notes, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(complete=True, candidate_id=args.candidate_id,
                         scientific_status=candidate["scientific_status"], stage=candidate["stage"],
                         retry_after=retry, disposition=disposition_ref,
                         notes=str(notes_path), queue_unchanged=True), indent=2))
finally:
    writer.close()
