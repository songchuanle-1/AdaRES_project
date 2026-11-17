"""Render the prospective design with the complete current goal and no gate claims."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "e636ce91a12e45488093b07fa2fc092b"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
goal_data = (ROOT / "pipeline/goal.json").read_bytes()
assert hashlib.sha256(goal_data).hexdigest() == "53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966"
goal = json.loads(goal_data)
review = """# Motivation Review - AdaRES_project
Tick e636ce91a12e45488093b07fa2fc092b. Autonomous, no human judgment implied.

## 1. Goal
Complete unchanged goal.json, SHA53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966:
```json
""" + json.dumps(goal, indent=2, ensure_ascii=True) + """
```
Target .515; historical W2 .5100/gap .0050; Phase1 unmet/Phase2 unapproved.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| W2 competition confidence | champion, not causal proof | frozen full prior W2 .510007498 | target unmet |
| Word-evidence confidence Y35 | native training done/full primary .504801214 | current two-pass9508/141 | fresh paired harm/cost under evaluation |
| Local-adjacency Y36 | training preserved | current snapshot22/30/rounded .5099 | trained frozen results unknown |
| Global pair-distribution Y37 | CPU information separation | current CPU32/equal moment test | all native hard gates pending |

## 3. Experiment Assessment
| Experiment | mIoU | Motivation relation | Contribution |
|---|---|---|---|
| B1 | prior full .502111562 | faithful standard indicator | baseline, not today's comparator yet |
| W2 | prior full .510007498 | learned competition | current champion, fresh repeated comparison pending |
| Y33 | prior full .505137821 | visual token attention | small/W2 harm already rejected; do not reopen |
| Y35 | today's first .504801214/repeat .504802891 | FG/BG to word attention | target unmet, scientific disposition pending paired intervals/cost |
| Y31 | prior full .505514618 | covariance confidence | inconclusive, no promotion; not a blanket geometry veto |

## 4. Gap Analysis
Current final-head competition/covariance does not explicitly compute weighted
global pair-distance distributions. Upstream spatial attention can encode
related information; no global blindness is asserted. Y37's CPU counterexample
distinguishes its direct readout from actual W2/Y31/Y36 classes on one exact
equal-moment fixture. The uniform masks mean scene shape participates; native
mask-specific incidence and intervention controls are still mandatory.
GT oracle quality is diagnosis only, never an inference feature.

## 5. Design Rationale
Y37 tests whether final-support distribution supplies useful confidence
information beyond first/second moments and nearest-neighbor features.
Fixed Laplace orders1/2/3 are conservative structural choices, not a proven
optimal basis. No validation, numeric, radius or feature selection occurs.
Structural feasibility supports falsifiable training only; trained .515/harm/
paired cost decide benefit. A negative result closes this realization, not all
geometric evidence routes. No external novelty search was performed this tick;
novelty and semantic necessity remain unknown.

## 6. Next Batch
One admission maximum, only after six gates. See
pipeline/designs/exp_design_Y37_e636ce91.md. Preserve Y36GPU1.
"""
design = """# Experiment Design - AdaRES_project
Status: DRAFT BEFORE NATIVE GATES. Research Pipeline autonomous.
Tick e636ce91a12e45488093b07fa2fc092b; original deadline unchanged.

## 1. Motivation
"AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned evidence,
targeting ScanRefer mIoU >= 0.515."
Upstream: pipeline/reviews/motivation_review_Y37_e636ce91.md.
Capacity2/active1/budget1; Y35 terminal, preserve Y36GPU1.
Test final-support global distance-distribution evidence, not a numeric setting.

## 2. Current Best Results
| ID | State | mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|
| W2 | prior frozen champion | .510007498 | not loaded here | not loaded here | 30 |
| B1 | prior frozen baseline | .502111562 | not loaded here | not loaded here | 30 |
| Y35 | today's frozen primary | .504801214 | .559739167 | .468181191 | last30/best17 |
| Y36 | current snapshot training | rounded .5099 | unknown | unknown | monitor22/30 |
First Y35 pass primary, not better repeated value. Fresh comparison pending.

## 3. Analysis And Direction
CPU32 actual-class test: equal feature/mask/coordinate first/second moments,
different global pair-distance distribution; W2/Y31/Y36 direct readouts delta0,
Y37 .000635847449. Repeated/detached/padded output unchanged; invariance max
5.96046448e-8; all8 gradients, min .000827530515.
This is a counterexample to exact readout equivalence, not population benefit
or complete representational independence from the decoder. Existing attention
and closed Y33/Y28/Y30 adverse evidence remain. Native steps must run next.

## 4. Experiment Plan
| ID | Config | Purpose | Expected Result | Priority | Dependency |
|---|---|---|---|---|---|
| AdaRES-Y37 | configs_song/model_paper5/adares_y37_pair_distribution_res.yaml | final mask pair-distribution confidence | falsifiable frozen trained comparison; no promised gain | P0 | six gates |
Point-mass-weighted soft FG/BG draws over superpoint centroids, with replacement;
scene-normalized squared distance, exp(-k*d2/scale), k1/2/3 fixed. FGFG/BGBG/FGBG
3-vectors through shared3->256->256/GELU encoder; normalized final query,
encoded FGFG-BGBG and encoded FGBG ->768->256->2 final indicator residual.
New264194params/eighttensors; about.16% of164629637.
Parent fork_rng initialization protected; own sixfile package/config.
No masks/auxiliary losses/querycount/schedule/inference-argmax changes.
Prospective sampling and all control/resource tests frozen in
pipeline/analyses/Y37_realization_freeze_e636ce91.md.

Frozen post-training:full validation9508 expressions/141scenes, Y37/W2/B1 each
two passes, FIRSTprimary; mIoU/Acc25/Acc50/activequeries/latency/peakmemory.
Whole-scene clustered10000 bootstrap seed1999 vs both comparators; TRAIN raw
pointcuts1487/3325; subgroup material harm when CIupper<0 against either;
CIcross0 inconclusive, not equivalent/noharm. Measure repeat noise and trained
paired costs empirically. Promote only after target.515 and frozen harm/Pareto
review; no initialized probe substitutes for those outcomes.

## 5. GPU Allocation
| Server | GPU | Assignment | Conditions |
|---|---|---|---|
| server-2 | 1 | preserve original Y36 | no restart/migration/source edit |
| server-2 | 0 | Y35 frozen eval then Y37 probes | live idle/exclusive check before each GPU test |
Training only after genuine admission/canonical dispatcher/growing native steps.
No server6, other projects, scheduling edits or background decision process.

## 6. Implementation Checklist
Independent sixfile package/config copied from W2, final readout only.
CPU actual class32 completed. Native diagnosis/controls/backward/resources,
adverse-loss investigation/review/commit/exact-byte alignment/receipt/launch
PENDING here. Copied nonDEC files byte-identical to local parent; imports and
all parent classes audited in scoped additive upload. Native parent30epochs/
seed1999/releasedbackbone/native128/oneargmax/optimizer unchanged.
Commit only seven own candidate files; no unrelated staged changes; no push
claim without actual push evidence. Sync only authorized needed server2.

## 7. Risk Assessment
Quadratic pair kernel matrices may violate lightweight cost despite small
parameter count; actual native largestpair728/6780 required. Pre-measure budget:
newparams<1% W2, native peakreserved<24GiB/noOOM/nonfinite. Timing2warmups/3reads
reported, never presumed speedup/trained Pareto.
Soft superpoint weights are approximate rawpoint geometry; orders1/2/3 not a
unique theoretical optimum. Random head may harm native loss; investigate
without selecting settings or deploying temporary weights. Wrong controls can
learn, so learnability does not prove semantic necessity. Preserve failures.
Native sparse repeat/continuation noise and one training seed remain.
No validation tuning, GT inference, new data collection or relaxed gates.

## 8. Smell Check
| Signal | Judgment |
|---|---|
| <5K/no new graph | no:264194 and new all-pair mask-conditioned distribution aggregation |
| just norm/linear/gate | projections visible, but weighted nonlinear point-pair kernel graph defines new interactions |
| numeric-only | no: new final support geometry path, masks/parent numerical settings unchanged |
| sweep | no: inherited256, fixed structural three orders, no search; arbitrary optimum not claimed |
| renamed ban | no threshold/temperature/radius/occupancy/rankingmargin/feature search |
| better numeric config only | no: exact matched old-readout counterexample and wrong-pair control |
Nontrivial graph/inherited or conservative frozen constants/explicit causal
question; novelty and efficacy unknown. Three standards permit testing only.
"""
for relative, text in (
    ("pipeline/reviews/motivation_review_Y37_e636ce91.md", review),
    ("pipeline/designs/exp_design_Y37_e636ce91.md", design),
):
    path = ROOT / relative
    assert not path.exists()
    path.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps(dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
