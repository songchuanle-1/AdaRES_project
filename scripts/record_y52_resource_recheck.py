"""Record a due resource-only continuation without replaying completed tests."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path("C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
sys.path.insert(0, str(SKILL))
import closure
import foreground_tick as tick
import pipeline_monitor as monitor


def evidence(path):
    resolved = (ROOT / path).resolve()
    assert ROOT in resolved.parents and resolved.is_file(), path
    return {"path": path, "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--observed-start", required=True)
    parser.add_argument("--observed-end", required=True)
    args = parser.parse_args()
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None, "Pipeline writer busy"
    try:
        owner = tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")
        assert owner["token"] == args.token
        assert owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
        now = datetime.now(timezone.utc)
        assert (datetime.fromisoformat(owner["deadline_at"]) - now).total_seconds() > 180
        start = datetime.fromisoformat(args.observed_start)
        end = datetime.fromisoformat(args.observed_end)
        assert start.tzinfo and end.tzinfo and start <= end <= now
        assert 0 <= (now - end).total_seconds() < 1800
        assert args.token in args.resource
        resource = tick.load(ROOT / args.resource)
        assert resource["complete"] and resource["read_only"] and resource["no_GPU"]
        assert resource["idle_eligible_gpu_ids"] == []
        assert not resource["GPU_lease_acquired"] and resource["no_resource_reservation"]
        gpus = {row["gpu"]: row for row in resource["resources"]}
        assert set(gpus) == {0, 1}
        y50 = resource["Y50"]
        assert y50["same_process_confirmed"] and y50["native_steps_advanced"]
        assert not y50["crash_marker_in_last_32768_bytes"]
        assert y50["pid"] in gpus[1]["compute_pids"]
        assert gpus[0]["compute_pids"] and y50["pid"] not in gpus[0]["compute_pids"]
        assert all(not row["eligible_for_exclusive_diagnostic"] for row in gpus.values())
        context_path = ROOT / "pipeline/headless/.tick_context.json"
        assert hashlib.sha256(context_path.read_bytes()).hexdigest() == owner["context_sha256"]
        pending_path = ROOT / "pipeline/headless/.tick_context_state.next.json"
        assert hashlib.sha256(pending_path.read_bytes()).hexdigest() == owner["pending_sha256"]
        queue = tick.load(context_path)["queue"]
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        assert view["generated_at"] == owner["generated_at"]
        assert view["deadline_at"] == owner["deadline_at"]
        prior = tick.load(ROOT / "pipeline/analyses/postflight_failure_4567_4567f5a09ecb4aa89df8f0c2932652df.json")
        checker_hash = hashlib.sha256((SKILL / "closure.py").read_bytes()).hexdigest()
        assert checker_hash == prior["checker"]["sha256"], "Reassess changed checker"
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y52")
        assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
        assert candidate["required_server"] == "server-2"
        assert datetime.fromisoformat(candidate["retry_after"]) <= start
        for row in candidate["evidence"]:
            assert evidence(row["path"]) == row, row["path"]
        bundle_path = "pipeline/analyses/y52_prototype_bundle_4567_4567f5a09ecb4aa89df8f0c2932652df.json"
        for row in tick.load(ROOT / bundle_path)["files"]:
            assert evidence(row["path"]) == row, row["path"]
        assert closure.exclusive_gpu_slot_available(candidate, queue["servers"], queue["active_tasks"]) is True
        report_text = (ROOT / ("pipeline/reports/summary_" + args.token + ".md")).read_text(encoding="utf-8")
        metric = re.search(r"\[adares_y50_joint_incidence_res\] Best=([0-9.]+) (\d+)/(\d+)", report_text)
        assert metric
        best = float(metric.group(1))
        before, after = y50["progress_before"], y50["progress_after"]
        retry = (end + timedelta(hours=1)).isoformat()
        previous_retry = candidate["retry_after"]
        audit_path = "pipeline/analyses/y52_resource_recheck_" + args.token + ".json"
        notes_path = "pipeline/analyses/notes_" + args.token + ".json"
        assert not (ROOT / audit_path).exists() and not (ROOT / notes_path).exists()
        foreign = ",".join(str(pid) for pid in gpus[0]["compute_pids"])
        occupancy = (
            f"客户端{args.observed_start}至{args.observed_end}只读复核："
            f"GPU0外来PID{foreign}/{gpus[0]['utilization_percent']}%/{gpus[0]['memory_mib']}MiB；"
            f"GPU1本项目Y50 PID{y50['pid']}/{gpus[1]['utilization_percent']}%/{gpus[1]['memory_mib']}MiB；"
            "无空闲授权卡，未预约或取得GPU诊断租约。"
        )
        progress = (
            f"Y50同一进程两次独立原生读数Train[{after[0]}/{after[1]}]步"
            f"{before[2]}→{after[2]}/{after[3]}，实际训练进度已确认，无有限尾部崩溃标记。"
        )
        process_metrics = json.dumps(dict(point=y50['latest_rounded_validation'],
            superpoint=y50['latest_rounded_superpoint_validation']), ensure_ascii=False)
        audit = dict(
            complete=True, token=args.token, observed_start_local=args.observed_start,
            observed_end_local=args.observed_end, snapshot_at=view["generated_at"],
            execution_status="INCOMPLETE: canonical abort and independent release check pending",
            science="Y50 live training; SAME frozen Y52 remains exclusive-GPU blocked",
            resource=evidence(args.resource), prototype=evidence(bundle_path),
            checker_unchanged=checker_hash, known_queue_opening_rejecting_predicate=True,
            exact_prior_failed_check=prior["failed_check"], finish_invoked=False,
            previous_retry_after=previous_retry, next_retry_after=retry,
            native_forward_or_backward=False, new_candidates=0, new_CPU_integration_tests=0,
            model_changes=0, admissions=0, launches=0,
            limitation="Resource recheck only; previous CPU controls not repeated or counted as new science; native TRAIN/backward/resource/restore gates remain unrun.",
        )
        notes = dict(actions=[
            "执行状态INCOMPLETE，待规范abort核实释放；科学状态Y50训练进度已确认，同一冻结Y52仍等待独占GPU。未改目标、门禁、调度、训练协议或其他项目。",
            occupancy + f"候选回查条件更新为{retry}，不是已知自动化时间或空闲证明。",
            progress + f"begin最佳mIoU {best:.4f}；本轮stdout取整验证{process_metrics}。仅过程指标，不是项目最佳或收益证明；延迟和峰值显存未知。",
            "本轮只有一次到期资源核验，无新的研究结论。既有完整CPU构造/初态及组件控制不重复运行或记为新成果；原型七文件与既有绑定证据哈希均核对未变。新CPU测试0、模型改动0、新候选0、准入0、新启动0。",
            "只读核验已知queue-opening拒绝谓词仍为真且检查器字节未变：本项目队列有开口不代表外来GPU0可用。旧精确失败为CLOSURE_INCOMPLETE: Blocked GPU candidate has a queue opening。本轮未重放已知失败finish，不伪造队列或阻塞类别，不改检查器。",
            "复用begin生成的0至3段事实表；本段实时观察不回写成preflight事实。收据用量仅begin到receipt的本地可见窗口，当前收据尚未生成、实际用量未知；缓存含在输入、推理含在输出，不重复累加或当账单。实际产出为本次资源证据与真实阻塞/收尾记录，非训练效果；释放后不写项目。",
        ], decisions=[
            "保留同一稳定Y52和冻结实现，按真实资源观察延后回查；空闲授权卡经实时检查且取得独占租约后，续做实际稀疏TRAIN匹配、前后向、共享query梯度、资源与完整恢复。未执行项不作PASS，Y50提交许可不扩展为Y52提交许可。",
            "Y50继续released SPFormer/seed1999/固定30/128queries/singleargmax，不停止、移动、重启或用验证调参；外来GPU0任务不操作。",
            "由于现有收尾谓词不能接受真实外来占用，保存证据后规范abort并只读核实释放；本轮不称完成。已有阻塞与过程训练进度没有新的可行动变化，不反复通知或新增审批；下一实际自动化时间未知。",
        ])
        memory_path = ROOT / "pipeline/codex_memory.md"
        lines = memory_path.read_text(encoding="utf-8").splitlines()
        old_token_match = re.search(r"^- 当前token：([a-f0-9]{32})", "\n".join(lines), re.MULTILINE)
        assert old_token_match and old_token_match.group(1) != args.token
        previous_token = old_token_match.group(1)
        previous_receipt = tick.load(ROOT / ("pipeline/tick_receipts/" + previous_token + ".json"))
        previous_status = previous_receipt["status"]
        previous_recorded = previous_receipt["recorded_at"]
        q = view['queue']
        r = view['research']
        replacements = [
            (("- 当前token",), f'- 当前token：{args.token}；快照{view["generated_at"]}；不可改deadline{view["deadline_at"]}；{view["mode"]}/续做同一Y52。'),
            (("- 本轮INCOMPLETE",), "- 本轮INCOMPLETE：到期资源核验仍无空闲授权卡，已知收尾谓词未改变；真实证据保存后规范abort，尚待核实释放，训练未停止。"),
            (("- 本轮begin监控",), f'- 本轮begin监控Y50 best {best:.4f}、{metric.group(2)}/{metric.group(3)}；本轮stdout取整{process_metrics}；仅过程验证，不是项目最佳或收益证明。'),
            (("- Y52仍blocked",), f"- Y52仍blocked/exclusive_gpu_slot/server-2；到期真实资源核验后retry为{retry}，计时器不是空闲证据。原型无改动/实现修订/提交/部署；原生匹配/反向与资源恢复仍待。"),
            (("- 本轮begin17:", "- 本轮队列"), f'- 本轮队列快照{view["generated_at"]}：active{q["active_count"]}/running{q["running_count"]}/claim{q["claiming_count"]}/wait{q["waiting_count"]}/target{q["target_active"]}/准入预算{q["candidate_budget"]}/储备{r["reserve_count"]}of{r["reserve_target"]}/准备预算{r["reserve_budget"]}；本轮新CPU测试0、模型改动0、新候选0、准入0、启动0。'),
            (("- 本轮真实客户端",), "- 本轮真实" + occupancy),
            (("- Y50原log",), "- Y50原log train_adares_y50_joint_incidence_res_20261016_145628.log；" + progress + "released SPFormer/seed1999/fixed30/128query/singleargmax不变。"),
            (("- 上轮44f779", "- 上轮执行"), f"- 上轮执行{previous_token}，收据{previous_status}，记录于{previous_recorded}；不重开旧token。pending为释放前字段，独立释放核验以原轮工具结果为准。"),
            (("- 本轮唯一begin",), "- 本轮唯一begin --auto-steady后先读compact/按ID续Y52。到期仅复查真实资源及两次原生进度；不重复既有CPU构造/组件测试，不为储备凑候选。"),
            (("- 本轮证据",), f"- 本轮证据{audit_path}；六段报告留档。只读核验既知拒绝谓词后不重放finish，规范abort后只读核实；释放后不写项目。"),
        ]
        for prefixes, replacement in replacements:
            indexes = [i for i, line in enumerate(lines) if line.startswith(prefixes)]
            assert len(indexes) == 1, prefixes
            lines[indexes[0]] = replacement
        assert len(lines) < 120 and sum(line.startswith("## ") for line in lines) == 7
        # Validate every input and memory anchor before the first mutation.
        tick.save(ROOT / audit_path, audit)
        candidate["retry_after"] = retry
        candidate["last_resource_check_at"] = args.observed_end
        candidate["blocker"] = occupancy + "既有CPU构造与组件对照已完成；下一项原生TRAIN/反向/资源恢复需要真正独占授权GPU。"
        seen = {row['path'] for row in candidate['evidence']}
        for path in (args.resource, audit_path, "scripts/record_y52_resource_recheck.py"):
            if path not in seen:
                candidate["evidence"].append(evidence(path))
        work["design_cycle"] = dict(stop_reason="external_resource_blocked",
            attempted_ids=["AdaRES-Y52"], evidence=[evidence(args.resource), evidence(audit_path)],
            resume_condition=candidate["unblock_condition"], retry_after=retry)
        tick.save(ROOT / "pipeline/research_work.json", work)
        memory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tick.save(ROOT / notes_path, notes)
        print(json.dumps(dict(token=args.token, notes=str(ROOT / notes_path),
            evidence=evidence(audit_path), next_retry_after=retry,
            new_CPU_integration_tests=0, admissions=0, launches=0,
            next_action="canonical report then abort and verify release"), ensure_ascii=False, indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
