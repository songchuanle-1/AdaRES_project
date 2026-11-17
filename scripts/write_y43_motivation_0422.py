"""Preserve the complete goal and bounded evidence upstream of admission."""

import argparse
from datetime import datetime, timezone
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
goal = json.loads((ROOT / "pipeline/goal.json").read_text())
path = ROOT / "pipeline/reviews/motivation_review_Y43_0422d0d8.md"
assert not path.exists()
text = "# Motivation Review: AdaRES-Y43\n\n## 1. Goal\n```json\n"
text += json.dumps(goal, indent=2) + "\n```\n"
text += """
Target.515; Phase1 unmet/Phase2 unapproved. W2 retained; rounded CSVY36
.5110/gap.004 is not promotion. GPU0/design explicit human grant retained.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Metric-aligned confidence improves selection | unconfirmed | historical frozen W2 comparisons | trained paired uncertainty |
| Explicit covariance captures distribution asymmetry | disproved for matched fixture only | actual32-seed reflection existing readouts0/thirdgap>=4 | nonnegative fixture/native task relevance |
| Third-moment confidence helps segmentation | untested | stableY43 preregistration | native feasibility and fixed30 trained evaluation |

## 3. Experiment Assessment
| Experiment | mIoU | Motivation Relation | Contribution |
|---|---|---|---|
| Y39 | historicalFIRST .50770598 | marginal heterogeneity | complete_inconclusive, no superiority |
| Y41 | original metadata .509053826 | mask-boundary probability | terminal/frozen review ongoing; not a Y43 acceptance criterion |
| Y42 | structured current unknown | full joint covariance | preserve original GPU1 fixed30 training |

## 4. Gap Analysis
All explicit first/second-order point summaries can be unchanged while
asymmetry differs. The first actual-class witness preserves32 channels,
mask/query/geometry/counts; its negative-feature caveat is retained.
Nonnegative confirmation and native incidence/interventions are executable
work, not missing-evidence blockers. Task benefit and upstream encoding unknown.

## 5. Design Rationale
One stable Y43 independent W2 package adds all32 third central moments,
stable signed cube-root and detached-mask point-mass FG/BG final confidence.
No tuning/feature subset/scalar/new loss; dimensions/defaults inherited.
All wrong/reflection/repeat/nonindicator/gradient/resource outcomes retained.
No novelty, necessity or human approval claims.

## 6. Next Batch
Exactly Y43 within budget1. See exp_design_Y43_0422d0d8.md seven sections
plus explicit smell2/3 defense. Every native gate remains pending until
executed. Training only after actual six-gate receipt; terminal full9508/141
twice EACH/FIRST/scenes10000seed1999/size/object/harmvsEither/noise/trainedcost.
"""
path.write_text(text, encoding="utf-8")
print(str(path))
