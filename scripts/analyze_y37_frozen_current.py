"""Reuse frozen FIRST-pass scene bootstrap and corrected empirical repeat noise."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

from analyze_y27_frozen_terminal import comparison

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "5e172bd47ee3450dbee928bdecaede7f"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
path = ROOT / ("pipeline/analyses/y37_frozen_full_retry_" + TOKEN + ".json")
value = json.loads(path.read_text())
assert value["complete"]
captures = value["rows"]
assert set(captures) == {"y37", "w2", "b1"}
assert all(c["complete"] and c["completed_passes"] == 2 for c in captures.values())
result = {
    "complete": True, "id": "AdaRES-Y37",
    "source": {"path": path.relative_to(ROOT).as_posix(),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    "observed_at_local": datetime.now().astimezone().isoformat(),
    "primary": "FIRST complete pass, not favorable repeat",
    "bootstrap": {"draws": 10000, "seed": 1999, "unit": "whole scene"},
    "summary": {k: c["passes"][0]["summary"] for k, c in captures.items()},
    "repeated_miou": {k: c["passes"][1]["summary"]["miou"] for k, c in captures.items()},
    "versus_w2": comparison(captures["y37"], captures["w2"]),
    "versus_b1": comparison(captures["y37"], captures["b1"]),
    "target": .515, "target_reached": captures["y37"]["passes"][0]["summary"]["miou"] >= .515,
    "cost_first_chunk_only": {
        k: {"parameters": c["parameters"], "median_seconds": statistics.median(c["warm_first_chunk_seconds"]),
            "peak_allocated_gib": c["peak_allocated_gib"], "peak_reserved_gib": c["peak_reserved_gib"],
            "repeat_max_iou_difference": c["repeat_max_iou_difference"],
            "repeat_miou_difference": c["repeat_miou_difference"]}
        for k, c in captures.items()
    },
    "limitations": ["Official unique/multiple metadata unknown.",
                    "Two repeats are empirical noise, not universal error or training-seed uncertainty.",
                    "First-chunk cost is not trained paired Pareto evidence; separate cost probe required.",
                    "No novelty, upstream blindness, equivalence, semantic necessity or no-harm claim."],
}
output = ROOT / ("pipeline/analyses/y37_clustered_comparison_" + TOKEN + ".json")
assert not output.exists()
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
