"""Persist only this tick's scoped candidate evidence under its live lease."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "8fd5e469dee241da8f21da7c870c0d3f"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor


def evidence(relative):
    path = ROOT / relative
    assert path.is_file()
    return dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def attach(candidate, relative):
    entry = evidence(relative)
    previous = next((e for e in candidate["evidence"] if e["path"] == relative), None)
    assert previous is None or previous == entry, "Never rewrite a bound artifact"
    if previous is None:
        candidate["evidence"].append(entry)


lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    y36 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y36")
    for relative in (
        "pipeline/analyses/Y36_realization_freeze_8fd5e469.md",
        "pipeline/reviews/motivation_review_Y36_8fd5e469.md",
        "pipeline/designs/exp_design_Y36_8fd5e469.md",
        "scripts/probe_y36_structure_cpu.py",
        "scripts/probe_y36_structure_cpu_v2.py",
        "scripts/y36_edge_readout.py",
        "scripts/build_y36_probe_protocols.py",
        "scripts/resolve_y36_cpu_fixture.py",
        "pipeline/analyses/y36_structure_cpu_" + TOKEN + ".log",
    ):
        attach(y36, relative)
    completed = []
    for name in ("source_bundle", "probe_protocols", "structure_cpu_v2",
                 "edge_diagnosis", "native_controls_v2", "physical_native", "resources",
                 "initial_loss"):
        relative = "pipeline/analyses/y36_" + name + "_" + TOKEN + ".json"
        path = ROOT / relative
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            assert result["complete"]
            attach(y36, relative)
            completed.append(name)
    y36["stage"] = "validation"
    y36["config"] = "configs_song/model_paper5/adares_y36_local_adjacency_res.yaml"
    missing = [n for n in ("edge_diagnosis", "native_controls_v2", "physical_native", "resources")
               if n not in completed]
    y36["next_action"] = (
        "Continue SAME fixed local-adjacency realization: pending actual native tests "
        + ",".join(missing)
        + ". Review adverse losses with matched fixed temporary-head investigation if needed. "
        "Only after substantive six-gate/smell review commit seven own files, verify deployed bytes, "
        "admit and canonical-dispatch on live-idle GPU1; preserve original Y35 GPU0. "
        "CPU32 equal-marginal edge delta.005216168, repeat0/perm6.71e-8/eight gradients; "
        "original float-mean fixture failure preserved, exact dyadic fixture fixed without tolerance/model changes. "
        "No trained efficacy, necessity, novelty or Pareto claim."
    )
    y33 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y33")
    for name in ("terminal", "y33_full_eval", "w2_full_eval", "b1_full_eval",
                 "clustered_comparison", "trained_cost", "postaudit"):
        relative = "pipeline/analyses/y33_" + name + "_" + TOKEN + ".json"
        if (ROOT / relative).exists():
            attach(y33, relative)
    if y33["stage"] != "rejected":
        y33["stage"] = "validation"
        y33["next_action"] = (
        "SAME Y33 canonicaldone_complete2026-10-09 17:47:55, originalPID64797 absent; "
        "last epoch30/best epoch20 metadata.5051367879, hashes unchanged. "
        "Frozen firstfull.505137821/repeat.505147631/9508expressions141scenes; target unmet. "
        "Complete fresh W2/B1 twice, original10000seed1999 whole-scene bootstrap/TRAINcuts1487/3325, "
        "trained paired cost and read-only checkpoint release audit before scientific closure; "
        "retain expression repeat max1 and mean9.809e-6, not global determinism. "
        "No training restart or Y33 source/checkpoint mutation; globalmarkerb49 is additive."
        )
    work_path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(y36_stage=y36["stage"], completed=completed,
                         y33_stage=y33["stage"], deadline=owner["deadline_at"])))
finally:
    lock.close()
