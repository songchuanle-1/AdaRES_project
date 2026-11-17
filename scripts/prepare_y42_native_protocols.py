"""Specialize fixed native protocols without changing samples or training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
marker = "79ce2de9582c369422facb27c3cbe85559bf227c"
records = []
for source_name, target_name in (
    ("upload_y39_test_bundle.py", "upload_y42_test_bundle.py"),
    ("probe_y39_native_controls.py", "probe_y42_native_controls.py"),
    ("probe_y39_physical_native.py", "probe_y42_physical_native.py"),
    ("probe_y39_resources.py", "probe_y42_resources.py"),
    ("probe_y39_initial_loss_investigation.py", "probe_y42_initial_loss_investigation.py"),
):
    source = ROOT / "scripts" / source_name
    text = source.read_text(encoding="utf-8")
    text = text.replace("67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773", marker)
    text = text.replace("adares_y39_point_distribution", "adares_y42_point_covariance")
    text = text.replace("MaskPointDistributionReadout", "MaskPointCovarianceReadout")
    text = text.replace("Y39", "Y42").replace("y39_", "y42_")
    text = text.replace("adares_y37_pair_distribution", "adares_y41_boundary_probability")
    text = text.replace("y37_", "y41_")
    if "native_controls" in source_name or "initial_loss" in source_name:
        text = text.replace("variance", "covariance")
        text = text.replace('native_covariance.shape[1] == 32', 'native_covariance.shape[1] == 528')
        text = text.replace(' and native_covariance.min() >= 0', '')
        anchor = "        return q, masks, pad, mass, changed"
        assert text.count(anchor) == 1
        text = text.replace(anchor,
            "        diagonal = torch.tensor([sum(32-j for j in range(i)) for i in range(32)], device=covariance.device)\n"
            "        changed[:, :, diagonal] = covariance[:, :, diagonal]\n" + anchor)
    if "native_controls" in source_name:
        text = text.replace("zero_covariance", "diagonal_only")
        anchor = "        return q, masks, pad, mass, torch.zeros_like(covariance)"
        assert text.count(anchor) == 1
        text = text.replace(anchor,
            "        diagonal = torch.tensor([sum(32-j for j in range(i)) for i in range(32)], device=covariance.device)\n"
            "        diagonal_only = torch.zeros_like(covariance)\n"
            "        diagonal_only[:, :, diagonal] = covariance[:, :, diagonal]\n"
            "        return q, masks, pad, mass, diagonal_only")
    compile(text, str(ROOT / "scripts" / target_name), "exec")
    target = ROOT / "scripts" / target_name
    assert not target.exists()
    target.write_text(text, encoding="utf-8", newline="\n")
    records.append({"path": target.relative_to(ROOT).as_posix(),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "parent_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
output = ROOT / ("pipeline/analyses/y42_protocol_bindings_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps({
    "complete": True, "id": "AdaRES-Y42", "files": records,
    "frozen_samples": "TRAIN first8 distinct scenes/init4/maxpair728,6780",
    "native_wrong_control": "Roll off-diagonal covariance only; keep all diagonal entries fixed.",
    "limitations": ["Protocol binding only, not an executed gate."],
}, indent=2), encoding="utf-8")
goal = json.loads((ROOT / "pipeline/goal.json").read_text(encoding="utf-8"))
review = ROOT / "pipeline/reviews/motivation_review_Y42_062ae665.md"
assert not review.exists()
review.write_text(
    "# Motivation Review: AdaRES-Y42\n\n## 1. Goal\n```json\n"
    + json.dumps(goal, indent=2, ensure_ascii=True) + "\n```\n"
    "Target .515; rounded CSV Y36 .5110/gap .004 is not promotion. W2 retained.\n\n"
    "## 2. Story Status\n"
    "| Hypothesis | Status | Evidence | Gap |\n|---|---|---|---|\n"
    "| Metric-aligned confidence improves selection | unconfirmed | prior frozen W2 comparisons | trained paired uncertainty |\n"
    "| Diagonal variance captures joint evidence | incomplete | actual Y42 CPU32 W2/Y39 output delta0, covariance delta2 | native occurrence/gradient/cost |\n\n"
    "## 3. Experiment Assessment\n"
    "| Experiment | mIoU | Relation to motivation | Contribution |\n|---|---|---|---|\n"
    "| Y37 | frozen .5050610 | geometry confidence | inconclusive versus W2; closed |\n"
    "| Y39 | original best metadata .5078108 | marginal point heterogeneity | full frozen review ongoing, not efficacy |\n"
    "| Y41 | training, raw .4697 at preflight7/30 | probability boundary evidence | preserve original GPU0 run |\n\n"
    "## 4. Gap Analysis\n"
    "All channel marginals can be identical while cross-channel co-occurrence differs. "
    "Actual readouts were tested, not conceptually assumed blind. Synthetic controls "
    "do not prove sparse-backbone blindness, semantic utility or necessity.\n\n"
    "## 5. Design Rationale\n"
    "Y42 adds the complete symmetric covariance sufficient statistic, with no channel "
    "subset, radius or temperature choice. Native wrong-covariance controls retain "
    "diagonal entries. Fixed samples, disabled-path equality, original native two-step "
    "entry/resume and paired cost determine feasibility. All adverse/incorrect-control "
    "outcomes must remain visible; no training-before-gate or diagnostic weights deployed.\n\n"
    "## 6. Next Batch\n"
    "Only Y42, budget1. Seven-section formal design and smell review before any "
    "admission. Keep seed1999/30epochs/released backbone/optimizer/loss/native128/one "
    "argmax and posttraining FIRST whole-scene protocol. No novelty claim or Phase2.\n",
    encoding="utf-8", newline="\n",
)
print(json.dumps({"files": len(records), "protocols": str(output), "motivation": str(review)}))
