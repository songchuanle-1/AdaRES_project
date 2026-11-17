# Motivation Review - AdaRES_project

Owned snapshot2026-10-06T10:51:41.101902Z. Review prepared after stable
diagnosis creation and actual Y29/Y30 tests, before full experiment design.

## 1. Goal

Goal description: "AdaRES: improve a faithful standard-indicator 3D referring
expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."

- Metric name=mIoU, target=.515, dataset=ScanRefer, baseline=AdaRES-B1.
- Constraints: server2 GPUs0/1 only, never server6; postB3 exactly one GPU,
  at most one independent job/card; no successor during B3, at most two
  structurally independent single-GPU candidates after its terminal state;
  B2/B4 fixed30 extension resumes epoch20 model/optimizer/scheduler without
  validation epoch tuning; released SPFormer sparse-UNet backbone with
  project decoder; report mIoU/Acc25/Acc50/queries/latency/peak memory;
  no validation thresholds/temperatures/weights/features/epoch tuning.
- Bans: no server6; no released b005/CFNet reported as B0; no validation
  ranking-margin or occupancy-weight scan.
- Pipeline policy: research_on_capacity_gap=true, steady_ticks_before_design1,
  fill_to_capacity=true, design_buffer0, max_candidates_per_design_tick2.
- Latest user/SERVERS override is stricter: GPU1 only, GPU0 reserved. Canonical
  snapshot budget1/active0/due0. B3 is historically terminal. Do not replay
  already fulfilled Y18/Y23 resume instructions.
- W2historical.5100, B1.5020, gap.0050, Phase1NOTYET, Phase2not authorized.
  Exact goal.json is hash-bound before admission; this review never rewrites it.

## 2. Story Status

| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Final confidence can lose candidate quality | Partial diagnosis | Native TRAIN64/256,163 oracle gaps | Cause and trained remedy unknown |
| Mask semantic readback rescues confidence | Rejected realization | Frozen Y28 small-v-W2 CI entirely negative | No universal readback ban |
| Pairwise regret adds a new objective here | Rejected Y29 | Actual B0 loss/gradient identity32cases | Not a blanket objective-family ban |
| Explicit ordered decoder trajectory supplies evidence | Diagnosis | Y30aligned7states, distinct trajectories163/163 | Native causality/resources/trained efficacy pending |

## 3. Experiment Assessment

| Experiment | mIoU | Motivation Relation | Contribution |
|---|---|---|---|
| W2 | .510107995 frozen reference | Metric-aware competition baseline | Champion, goal unmet |
| B1 | .502223014 latest frozen reference | Standard indicator baseline | Comparator retained |
| Y28 | .505520490 | Final mask visual content readback | Small2816/124 harm CI[-.032367651,-.003092544]; rejected |
| Y27 | .509987856 | Relative peer geometry | Whole-scene CIs cross0; inconclusive |
| Y29 | Untrained | Coupled regret objective | CPU actual-function0loss/0gradient difference; duplicate rejected |

Y28's all-v-W2 CI crosses0 and positive medium-v-B1 interval are preserved;
the frozen small-group harm rule still rejects that realization. No trained
result here establishes causality, seed robustness or lightweight Pareto gain.

## 4. Gap Analysis

Actual W2 head directly sees final query, overlap context and point mass, not
an ordered sequence. Upstream histories are implicit. The native seven query
states share sample identity and differ (mean historical-final RMS.83523146);
mask transition descriptors vary. TRAIN64 scene-mean oracle gap.154148389 is
not an deployable oracle selector or efficacy estimate.

The prior Y9 fixed score mean and B4 previous-layer heuristic are distinct
from the proposed trainable state-transition readout, but their negative
evidence remains. No broad claim of temporal necessity or novelty is made.

## 5. Design Rationale

Y30 adds a genuine ordered interaction across previously unexposed states,
not a LayerNorm-only intervention, coefficient scan, fixed score average,
margin change, mask-content readback or selection rule adjustment. GRU width
is inherited d_model256, sequence length comes from native seven states and
two descriptors encode soft mask overlap/log-area change. No width sweep.

CPU actual candidate32cases: wrong/reverse/terminal histories separate;
repeat/padding0, permutations below1e-5; all8 new tensors active. Native
controls and trained outcomes must determine whether this path is useful.
No literature search was completed; novelty is unknown.

## 6. Next Batch

See pipeline/designs/exp_design_Y30_86766fcd.md. Only stable Y30 may advance
through native controls/two-step/resume/resources/independence/constraints;
budget permits at most one new training admission. Do not stop at a draft.
