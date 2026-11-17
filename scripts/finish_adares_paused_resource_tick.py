"""Record unchanged observations while preserving the original resource pauses."""

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y102_deployment import git, Y102
import closure


SKILL = "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline"
OWNER = "01a07e32-78bf-7670-925b-370911ea48db"
SERVER1_PAUSE = (
    "pipeline/authorizations/"
    "server1_pause_20261115_ce356360782a431ab9adc3e015f7d5aa.json"
)
GPU1_PAUSE = "pipeline/authorizations/server2_gpu1_pause_20261104.json"
SERVER1_IDS = {
    "AdaRES-" + suffix
    for suffix in (
        "Y62", "Y63", "Y64", "Y66", "Y69", "Y72",
        "Y76", "Y79", "Y82", "Y84", "Y88", "Y103",
    )
}
GPU1_IDS = {"AdaRES-" + suffix for suffix in ("Y67", "Y70", "Y73", "Y75", "Y78", "Y81")}
FAILED_IDS = {"AdaRES-Y86", "AdaRES-Y97"}


def evidence_text(entry):
    return entry["path"] + " SHA" + entry["sha256"]


def record(token):
    lease(token, 180)
    assert os.environ["CODEX_THREAD_ID"] == OWNER
    view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
    snapshot = tick.load(ROOT / "pipeline/headless/.tick_context.json")
    assert view["mode"] == "event" and view["preflight"]["goal_rc"] == 4
    assert not view["changes"]["new_terminal_ids"]
    assert view["queue"]["candidate_budget"] == 0
    assert view["research"]["creation_budget"] == view["research"]["reserve_budget"] == 0

    paths = [
        "pipeline/goal.json", "AGENTS.md", "SERVERS.yaml",
        "pipeline/queue.json", "pipeline/running.json", "pipeline/experiments.csv",
        "pipeline/research_work.json", "pipeline/closure_policy.json",
        "pipeline/execution_mode.json",
    ]
    paths.extend(str(path) for path in (ROOT / "pipeline/admissions").glob("*.json"))
    protected = {path: ref(path) for path in paths}
    index = git("ls-files", "--stage", "-z")
    assert protected["pipeline/goal.json"]["sha256"] == (
        "4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a"
    )
    assert git("rev-parse", "HEAD").decode().strip() == Y102
    servers = monitor.load_servers(str(ROOT))
    assert [servers[name]["allowed_gpus"] for name in ("server-1", "server-2", "server-6")] == [
        [], [0], [],
    ]
    server1 = ref(SERVER1_PAUSE)
    gpu1 = ref(GPU1_PAUSE)
    assert server1["sha256"] == "856914c2f463b590eb615c4de9cc7168b2328087462fa56756c55523970be3e0"
    assert gpu1["sha256"] == "3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45"

    raw_path = ROOT / ("pipeline/checks/production_adares-y102_" + token + "_start.log")
    raw = tick.load(raw_path)
    assert raw["complete"] and raw["process_identity_verified"] and raw["pid"] == 4747
    assert raw["source_marker"] == Y102 and raw["source_files_verified"] == 6
    assert raw["native_YAML_semantics_verified"] and raw["no_bounded_tail_error"]
    assert raw["native_progress_confirmed"] is False
    assert raw["before"]["latest"] == raw["after"]["latest"] == dict(
        phase="Train", epoch=1, total=30, step=7, count=4686
    )
    assert raw_path.stat().st_mtime >= datetime.fromisoformat(view["generated_at"]).timestamp()
    assert raw["authorized_gpu_uuid"] + ", 4747" in raw["compute_owners"]
    assert raw["gpu"].split(",")[-1].strip() == "280.00"

    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {candidate["id"]: candidate for candidate in work["candidates"]}
        due_ids = snapshot["research_work"]["due_ids"]
        assert set(due_ids) <= SERVER1_IDS | GPU1_IDS | FAILED_IDS | {"AdaRES-Y57"}
        dispositions = []
        for cid in due_ids:
            candidate = candidates[cid]
            assert candidate["stage"] == "blocked"
            if cid == "AdaRES-Y57":
                assert candidate["blocker_kind"] == "human_authorization"
                reason = "separate_Y57_user_pause_unchanged"
                entries = candidate["evidence"][-2:]
            elif cid in FAILED_IDS:
                assert candidate["blocker_kind"] == "numerical_control_failure"
                assert candidate["failure_scope"] == "implementation"
                reason = "original_numerical_failures_unchanged_no_replay"
                entries = candidate["evidence"][-2:]
            elif cid in SERVER1_IDS:
                assert candidate["blocker_kind"] == "human_authorization"
                assert candidate["required_server"] == "server-1"
                assert candidate["dispatch_hold"] and candidate["resume_requires_user_authorization"]
                reason, entries = "latest_server1_user_pause_unchanged", [server1]
            else:
                assert candidate["blocker_kind"] == "human_authorization"
                assert candidate["required_server"] == "server-2" and candidate["required_gpu"] == 1
                reason, entries = "original_GPU1_user_pause_unchanged", [gpu1]
            for entry in entries:
                closure.artifact(ROOT, entry)
            dispositions.append(dict(id=cid, disposition=reason, evidence=entries))

        queue = monitor.load_queue(str(ROOT))
        assert queue["gpu_queues"]["server-1"] == []
        assert monitor.load_running(str(ROOT)) == {"server-2": {"0": "AdaRES-Y102"}}
        paused = next(row for row in queue["gpu_queues"]["server-2"] if row["id"] == "AdaRES-Y57")
        assert paused["status"] == "paused" and paused["dispatch_hold"]
        assert paused["resume_requires_user_authorization"]
        y88 = candidates["AdaRES-Y88"]
        archive = tick.load(closure.artifact(ROOT, y88["native_review_pause_archive"]))
        assert archive["native_captures_preserved"] == 8 and archive["cost_not_started"]
        assert not y88.get("frozen_trained_cost_job")
        assert y88["no_prediction_replay_after_reauthorization"]

        out = ROOT / ("pipeline/checks/unchanged_paused_resource_observation_" + token + ".json")
        assert not out.exists()
        tick.save(out, dict(
            complete=True, token=token, snapshot_generated_at=view["generated_at"],
            recorded_at=datetime.now(timezone.utc).isoformat(), material_change=False,
            actual_observation=raw, raw_observation=ref(raw_path),
            native_progress_confirmed=False, remote_observation_times_preserved=True,
            due_dispositions=dispositions, server1_pause=server1, GPU1_pause=gpu1,
            Y88_archive=y88["native_review_pause_archive"], Y88_cost_not_started=True,
            new_scientific_IDs=0, new_admissions=0, new_training_launches=0,
            new_GPU_tests=0, new_review_jobs=0, new_cost_jobs=0, scientific_replays=0,
            no_checkpoint_loaded=True, no_restart_or_migration=True,
            no_server1_connection=True, no_repeated_attachment_or_administrator_question=True,
            research_work_unchanged=True, no_new_user_action=True,
            evidence=[ref(__file__), ref(raw_path), server1, gpu1, y88["native_review_pause_archive"]],
        ))
        memory = ROOT / "pipeline/codex_memory.md"
        previous = memory.read_bytes()
        backup = ROOT / ("pipeline/snapshots/codex_memory_before_" + token + ".md")
        assert not backup.exists()
        backup.write_bytes(previous)
        assert backup.read_bytes() == previous
        old_lines = previous.decode("utf8").splitlines()
        lines = old_lines[:old_lines.index("## Last Tick")]
        updates = {
            "- Current token ": (
                "- Current token " + token + "; snapshot " + view["generated_at"]
                + "; deadline " + view["deadline_at"]
                + "; never extend. Latest server1/GPU1/Y57 user pauses remain authoritative."
            ),
            "- Exact previous seven-section memory ": (
                "- Exact previous seven-section memory " + evidence_text(ref(backup))
                + "; original science/receipts/negative evidence/terminal products/dedicated "
                "recovery/pause and automation history retained through exact-byte backup chain."
            ),
            "- Original Y102 PID": (
                "- Original Y102 PID4747/server2GPU0/280W retained; all4 native samples remain "
                "Train1/30 step7/4686; current progress NOT confirmed. Exact source6/nativeYAML/"
                "private runtime/physical compute observed " + evidence_text(ref(out))
                + ". Preserve actual remote times/logs/receipt/protocol; no restart/checkpoint load."
            ),
            "- Current allowed resources:": (
                "- Current allowed resources: server1[]/server2[0]/server6[]. Latest server1/GPU1/"
                "Y57 pauses remain. Original GPU0 UUID/PID4747/280W observed "
                + evidence_text(ref(raw_path))
                + ". GPU1 foreign process untouched; no server1 connection or resource substitution."
            ),
        }
        lines = [
            next((value for prefix, value in updates.items() if line.startswith(prefix)), line)
            for line in lines
        ]
        lines.extend([
            "## Last Tick",
            "- " + token + ": event; " + str(len(dispositions))
            + " due holds rechecked against unchanged user pauses and original numerical failures. "
            "Creation/admission/reserve budgets0, no terminal or operational change. Original Y102 "
            "source6/YAML/private runtime/280W confirmed, counter still Train1step7; no new science.",
            "- Y88 original8 captures/cost-unstarted and device bindings preserved. No new candidates/"
            "admissions/training/GPUtests/reviews/cost or CPU/native/terminal/prediction replay. No "
            "source/data/seed/power/scheduler/goal/receipt/ledger changes, restart, checkpoint load, "
            "server1 access, repeated ptrace/admin request or foreign interference.",
            "- Actual observation " + evidence_text(ref(out))
            + ". No material change or new user action; quiet heartbeat. Exact7-section memory backup "
            "retained; official report/finish/receipt/release authoritative.",
        ])
        assert len(lines) < 120 and sum(line.startswith("## ") for line in lines) == 7
        memory.write_text("\n".join(lines) + "\n", encoding="utf8", newline="\n")
        notes = ROOT / ("pipeline/reports/notes_" + token + ".json")
        assert not notes.exists()
        tick.save(notes, dict(
            actions=[
                "只读核验原Y102 PID4747的源码6文件、原生YAML、私有环境及server2 GPU0/280W。"
                "四次原生样本仍停在Train1/30第7/4686步，持续推进未确认，无新增日志错误。",
                "到期原挂起复查共" + str(len(dispositions))
                + "项；既有用户暂停及原数值失败证据保持。Y88八遍原预测保留、成本未启动。",
            ],
            decisions=[
                "创建/准备/准入预算为零，无新终态。保持server1全GPU暂停、GPU1暂停及Y57单独暂停，"
                "不迁移、不重启、不加载检查点、不回放科学检查或重复管理员问题。原goal、收据和候选"
                "账本保持字节不变；无新增可行动事项，按原频率静默收尾。",
            ],
            evidence=[ref(out), ref(backup), ref(raw_path), server1, gpu1],
        ))
        assert all(ref(path) == entry for path, entry in protected.items())
        assert git("ls-files", "--stage", "-z") == index
        return out, notes
    finally:
        writer.close()


def finish(token):
    out, notes = record(token)
    for arguments in (
        ["report", "--token", token, "--notes", str(notes)],
        ["finish", "--token", token],
    ):
        result = subprocess.run(
            [sys.executable, "-B", SKILL + "/foreground_tick.py", "AdaRES_project"] + arguments,
            cwd=ROOT, timeout=600,
        )
        assert result.returncode == 0, "Preserve evidence and reconcile this owned lease"
    receipt = tick.load(ROOT / ("pipeline/tick_receipts/" + token + ".json"))
    assert receipt["status"] == "FINALIZED" and receipt["postflight"] == "PASS"
    assert receipt["context_committed"] and receipt["lease_release"] == "released"
    assert not (ROOT / "pipeline/headless/.foreground.lockdir").exists()
    assert not (ROOT / "pipeline/headless/.tick_context_state.next.json").exists()
    pending = ROOT / "pipeline/archive_pending"
    assert not pending.exists() or not any(path.is_file() for path in pending.rglob("*"))
    print(json.dumps(dict(observation=ref(out), material_change=False, lease_release="released")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    finish(parser.parse_args().token)
