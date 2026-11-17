"""Attach this tick's actual evidence without rewriting existing bound artifacts."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "e636ce91a12e45488093b07fa2fc092b"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor


def attach(candidate, relative):
    path = ROOT / relative
    assert path.is_file()
    entry = dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    prior = next((e for e in candidate["evidence"] if e["path"] == relative), None)
    assert prior is None or prior == entry
    if prior is None:
        candidate["evidence"].append(entry)


lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    y37 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y37")
    for relative in (
        "pipeline/analyses/Y37_realization_freeze_e636ce91.md",
        "pipeline/reviews/motivation_review_Y37_e636ce91.md",
        "pipeline/designs/exp_design_Y37_e636ce91.md",
        "scripts/y37_pair_readout.py", "scripts/probe_y37_structure_cpu.py",
        "scripts/probe_y37_pair_diagnosis.py", "scripts/run_adares_authorized_probe.py",
    ):
        attach(y37, relative)
    completed = []
    for name in ("local_bundle", "source_bundle", "probe_protocols", "structure_cpu",
                 "pair_diagnosis", "native_controls_v3", "physical_native", "resources",
                 "initial_loss", "source_bundle_lf", "structure_cpu_lf"):
        relative = "pipeline/analyses/y37_" + name + "_" + TOKEN + ".json"
        if (ROOT / relative).exists():
            result = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            assert result["complete"]
            attach(y37, relative)
            completed.append(name)
    if y37["stage"] not in ("training", "rejected", "blocked"):
        y37["stage"] = "validation"
        y37["config"] = "configs_song/model_paper5/adares_y37_pair_distribution_res.yaml"
        missing = [n for n in ("pair_diagnosis", "native_controls_v3", "physical_native", "resources")
                   if n not in completed]
        y37["next_action"] = (
            "Continue SAME fixed Y37 global-pair realization. Pending actual native "
            + ",".join(missing)
            + "; investigate adverse native results with fixed controls, never select settings. "
            "After all six-gate/smell review, commit only seven own files, align exact tested "
            "bytes, admit/canonical-dispatch GPU0 and verify growing native steps. "
            "Preserve original Y36GPU1. CPU32 equal-moment/readout counterexample "
            "Y37delta.000635847449/W2Y31Y36zero, all8finite gradients; not trained "
            "efficacy, necessity, novelty or lightweight Pareto."
        )
    y35 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y35")
    for name in ("terminal", "y35_full_eval", "w2_full_eval", "b1_full_eval",
                 "clustered_comparison", "trained_cost", "postaudit"):
        relative = "pipeline/analyses/y35_" + name + "_" + TOKEN + ".json"
        if (ROOT / relative).exists():
            attach(y35, relative)
    if y35["stage"] not in ("complete", "rejected"):
        y35["stage"] = "validation"
        y35["next_action"] = (
            "SAME Y35 canonical done_complete confirmed preflight2026-10-10 11:48:56; "
            "originalPID108331 absent, last30/best17, checkpoint6b30454c preserved. "
            "Frozen primary.504801213799/repeat.504802890898/9508expressions141scenes. "
            "Complete frozen fresh W2/B1 twice first-primary, scene-bootstrap10000 "
            "seed1999/TRAINcuts1487/3325/material subgroup harm/trained paired cost "
            "and checkpoint/job preservation audit before scientific disposition. "
            "No blind restart, promotion or global-determinism claim."
        )
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True)+"\n", encoding="utf-8")
    print(json.dumps(dict(y37_stage=y37["stage"], completed=completed, y35_stage=y35["stage"])))
finally:
    lock.close()
