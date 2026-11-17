"""Persist actual Y94/Y96 results before aborting an expired tick."""
import argparse
from datetime import datetime
import json
import os
from prepare_y60_stage_heads import ROOT, monitor, ref, tick


def record(token):
    owner = tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")
    assert owner["token"] == token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
    assert datetime.now().astimezone() > datetime.fromisoformat(view["deadline_at"])
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        from y96_deployment import Y95, git
        assert git("rev-parse", "HEAD").decode().strip() == Y95
        assert ref("pipeline/goal.json")["sha256"] == (
            "4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a")
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y96")
        assert candidate["stage"] == "validation" and candidate.get("native_smoke_result")
        assert not candidate.get("admission_receipt")
        assert monitor.load_running(str(ROOT)) == {"server-2": {"0": "AdaRES-Y95"}}
        done = monitor.load_experiment(str(ROOT), "AdaRES-Y94")
        assert done["status"] == "done_complete"
        helpers = [ref("scripts/" + name) for name in (
            "advance_y96.py", "y96_deployment.py", "y95_y96_terminal_contract.py",
            "prepare_y95_y96_terminal_binding.py", "test_y95_y96_terminal_binding.py",
            "test_y96_admission.py")]
        failure = ROOT / ("pipeline/checks/y96_unfinished_toolchain_" + token + ".json")
        assert not failure.exists()
        tick.save(failure, dict(complete=False, token=token, observed_at=datetime.now().astimezone().isoformat(),
            actual_native_result=candidate["native_smoke_result"],
            Y94_verified_canonical_archive=done["terminal_evidence"],
            draft_helpers=helpers, draft_tests_executed=False, binding_created=False,
            source_selected=False, admitted=False, launched=False,
            attempted_command="python -B scripts/prepare_y95_y96_terminal_binding.py --token " + token,
            observed_failure="lease(token,300) refused because less than300s remained; no helper tests or "
                "binding generation ran. The original deadline is now past; abort without further research.",
            next_action="Do not replay native or Y94 terminal. In a fresh owned tick at HEADY95, inspect and "
                "test the draft Y96 server1 admission/source tools and actual-native Y95 terminal binding, "
                "then select exactly seven tested additions, inherit complete original Y94 server1 protocol, "
                "admit and canonical dispatch with fresh idle/unreserved350W GPU0 and increasing counters."))
        candidate["next_action"] = (
            "Actual original Y96 server1 native PASS retained; never replay. Native result "
            + candidate["native_smoke_result"]["path"]
            + ". At fresh owned HEADY95 tick first review/test draft advance_y96.py/y96_deployment.py/"
            "y95_y96_terminal_contract.py/prepare_y95_y96_terminal_binding.py and associated tests; "
            "no draft test/binding/source/admission/dispatch has passed or executed. Then actual-native-tested "
            "Y95 lifecycle binding, exact source selection, full original Y94 server1/350W freeze/receipt, "
            "admission and canonical launch plus increasing counters. Both pauses and original Y95 preserved.")
        candidate["evidence"].extend(helpers + [ref(failure)])
        tick.save(ROOT / "pipeline/research_work.json", work)
        memory = ROOT / "pipeline/codex_memory.md"
        backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
        assert not backup.exists()
        original = memory.read_bytes()
        backup.write_bytes(original)
        old = original.decode("utf-8").splitlines()
        lines = []
        for line in old[:old.index("## Last Tick")]:
            if line.startswith("- Current token "):
                line = "- Current token " + token + "; snapshot " + view["generated_at"] + "; deadline " + view["deadline_at"] + "; EXPIRED, preserve and abort; never extend."
            if line.startswith("- Exact previous seven-section memory:"):
                line = "- Exact previous seven-section memory: " + ref(backup)["path"] + " SHA" + ref(backup)["sha256"] + "; every historical protocol/receipt/failure/ban/recovery requirement byte-preserved."
            lines.append(line)
            if line == "## Active Hypotheses":
                lines += [
                    "- CURRENT Y96 FIRST actual server1GPU0/350W original native YAML/TRAIN two finite backward/optimizer steps PASSED: "
                    + candidate["native_smoke_result"]["path"] + " SHA" + candidate["native_smoke_result"]["sha256"]
                    + "; losses[4.615510940551758,6.309134006500244], peak_reserved6759120896. No efficacy claim or native replay.",
                    "- Y96 remains validation, HEADY95 unchanged, NO admission/selection/training. Draft tools and tests in "
                    + ref(failure)["path"] + "; UNTESTED. prepare_y95_y96_terminal_binding.py refused lease reserve300 before mutations/tests; original deadline elapsed. Review/test in next fresh tick, not rerun native.",
                    "- Historical below claims that Y96 native is unexecuted or waits on Y94 are superseded ONLY by these current actual records; original science/programs/freezes remain immutable.",
                ]
            if line == "## Running And Queue":
                lines += [
                    "- CURRENT Y94 naturally finished and EXACT original20261109_111838 full Train/Val1..30 stable finite last30/best21/model/optimizer544/scheduler30-21 verified ONCE and canonically archived: "
                    + done["terminal_evidence"]["path"] + " SHA" + done["terminal_evidence"]["sha256"]
                    + "; trainingbest .5075652599334717 is NOT formal FIRST; never replay terminal/restart.",
                    "- CURRENT only canonical running=server2GPU0 original Y95 PID34484; start observation this tick Train24/30 step2725/4686. Server1GPU0 was physically idle350W after Y94 and native; fresh occupation mandatory before next use. No Y96 launch.",
                    "- Old below Y94-running/current-observer claims are historical. Y95 source remains original HEADY95; old exact terminal helper still valid until NEW actual-nativeY96 binding passes. Future full review inherits complete original Y94 server1 freeze, NOT server2 substitute.",
                ]
        lines += [
            "## Last Tick",
            "- " + token + ": Y94 fully verified/canonically archived; Y96 FIRST actual native server1 PASS. Y95 original training preserved. ScientificIDs0/admissions0/launches0; no FIRST/cost/promotion/GPU1/Y57 use.",
            "- Draft Y96 admission/deployment/Y95 additive lifecycle tools written but not tested, binding not generated. prepare helper reserve300 refused before changes/tests. Real deadline expired; stop research and abort INCOMPLETE, never extend or fabricate PASS.",
            "- Next fresh tick MUST preserve native result and Y94 archive without replay, validate draft tools then actual-evidence Y95 binding/full original server1 freeze/source/admission/canonical launch and increasing counters. Untouched Y98/Y99 still require new bindings. All original receipts/bans/protocols/negative evidence preserved.",
        ]
        assert len(lines) < 120 and sum(line.startswith("## ") for line in lines) == 7
        memory.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            "完成：Y94原运行自然fixed30结束，精确目录20261109_111838，完整Train/Val1..30、"
            "稳定有限last30/best21/model/optimizer544/scheduler30-21核验并canonical归档，原权重留远端。",
            "运行：原Y95进度本轮已确认Train24/30 step2725/4686；未停止、重启、迁移或改变原签名收据。",
            "新准备：Y96首次真实server1GPU0/350W原生YAML/TRAIN两步有限loss/backward/optimizer和新增路径通过，"
            "losses4.615510940551758、6.309134006500244；无科学有效性结论。",
            "新增准入/部署和Y95兼容工具仍是未测试草稿；准备器在lease reserve300检查前置失败，"
            "未执行测试、生成绑定、选择源码、准入或启动。本轮真实deadline已过，保留全部产物并abort/INCOMPLETE。",
        ], decisions=[
            "不延长deadline、不冒充工具通过；新鲜tick优先验证草稿，复用实际native结果及Y94归档，不重复执行。",
            "保留完整原server1/350W冻结协议，禁止用server2代替；GPU1全部使用和Y57独立暂停，禁止server6、"
            "跨项目、调度/功耗变更或干预外部任务。"
        ], evidence=[ref(failure), candidate["native_smoke_result"], done["terminal_evidence"], ref(backup)]))
        print(json.dumps(dict(notes=str(notes), partial=ref(failure), memory_lines=len(lines),
                              deadline_expired=True, next_operation="official report then abort")))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    record(parser.parse_args().token)
