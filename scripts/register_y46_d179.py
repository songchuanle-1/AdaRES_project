"""Register a genuine independent mask-topology diagnosis before route audit."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    assert not any(c["id"] == "AdaRES-Y46" for c in work["candidates"])
    prereg = ROOT / "pipeline/analyses/Y46_topology_prereg_d179.md"
    assert not prereg.exists()
    text = """# Y46 Prediction-Blind Mask Topology Diagnosis

- Stable ID AdaRES-Y46; hypothesis_id global-mask-component-coherence.
- Hypothesis: predicted-mask connected-component point masses may provide
  explicit global coherence evidence absent from the parent/local adjacency
  readouts and feature-distribution moments. This is untested, not novelty.
- Do not reopen Y43 asymmetry or Y31/Y36 under a new name. Audit their actual
  source; execute equal-local-summary/different-connectivity controls first.
  Reject this realization with executable duplication/no-op evidence.
- Inherit the exact existing local-adjacency neighbor construction/defaults
  from actual server2 source, if one exists; no new k/radius/threshold choice.
  Native hard mask threshold and original point mass remain unchanged.
- First test: source audit plus controlled regular graphs whose local degrees,
  point masses and constant-feature one-hop summaries agree, but global
  component masses differ. This is a CPU structural witness, not native efficacy.
- Native follow-up before GPU becomes occupied: same prediction-blind TRAIN8
  indices0,44,60,101,115,126,137,171/seed1999 W2 masks and actual SP geometry;
  record all queries/GT component incidence, joint permutations and counts.
  Do not choose scenes/graph settings/components from IoU outcomes.
- Conditional realization, only if diagnosis holds: explicit query/mask
  component aggregation into final confidence, independent package/config;
  native full backward, disable-parent, matched controls and resources required.
  Component statistics alone, synthetic gradients or captured inputs cannot admit.
- No validation tuning, gate changes, diagnostic weight deployment or training.
  Any eventual admission retains releasedSPFormer/seed1999/fixed30/singleGPU,
  full9508/141 twice EACH/FIRST, W2+B1/scene uncertainty/frozen harm/paired cost.
"""
    prereg.write_text(text, encoding="utf-8")
    evidence = dict(path=prereg.relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256(prereg.read_bytes()).hexdigest())
    work["candidates"].append(dict(
        id="AdaRES-Y46", stage="diagnosis", hypothesis_id="global-mask-component-coherence",
        hypothesis="Global predicted-mask component point masses may expose confidence information absent from fixed local adjacency and point-feature marginal summaries.",
        matched_control="Actual W2/Y31/Y36 sources, equal-degree constant-feature graph witnesses, native inherited-neighbor TRAIN masks, repeat/joint-permutation/literal component counts.",
        next_action="Audit exact parent/local-adjacency source and run equal-local-summary/different-global-connectivity CPU test; then actual prediction-blind native TRAIN8 component incidence.",
        scientific_status="registered_diagnosis_not_admitted", evidence=[evidence],
        pending_gates=["diagnosis", "controls", "native_backward", "resources", "independence", "constraints"],
        history=[dict(observed_at=datetime.now().astimezone().isoformat(), action="stable_diagnosis_registered_before_source_audit")]))
    path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(id="AdaRES-Y46", stage="diagnosis", evidence=evidence)))
finally:
    writer.close()
