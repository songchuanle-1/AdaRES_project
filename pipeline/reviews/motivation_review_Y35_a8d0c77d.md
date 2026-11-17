# Motivation Review - AdaRES_project

Foreground tick a8d0c77d45d142c0a6c46b163b956253; autonomous research, not human approval.

## 1. Goal

Exact goal description: "AdaRES: improve a faithful standard-indicator 3D
referring expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."
Goal identity 53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966.
Baseline AdaRES-B1; metric mIoU, target0.515, dataset ScanRefer.
Historical champion W2 .5100, baseline B1 .5020, remaining gap .0050.

Exact goal constraints:
1. "Use server-2 GPUs 0 and 1 only; never use server-6."
2. "All experiments launched after AdaRES-B3 must use exactly one GPU; run at most one independent experiment on GPU 0 and one on GPU 1."
3. "Do not design or queue a successor while AdaRES-B3 is running; after B3 reaches a terminal state, admit at most two structurally independent single-GPU candidates."
4. "Use a fixed 30-epoch protocol for the B2 and B4 extension. Resume both from their epoch-20 last checkpoints without resetting the optimizer or LR scheduler; do not tune the epoch count on validation."
5. "Train from the released SPFormer sparse-UNet backbone; the decoder is authored in this project."
6. "Report mIoU, Acc@0.25, Acc@0.5, active queries, latency, and peak memory."
7. "Do not tune thresholds, temperatures, loss weights, feature subsets, or the fixed 30-epoch protocol on ScanRefer validation."

Exact bans:
1. "Do not use server-6."
2. "Do not report the released b005/CFNet checkpoints as AdaRES-B0 results."
3. "Do not scan the ranking margin or the occupancy loss weight on validation."

Policy: research_on_capacity_gap true, steady_ticks_before_design1,
fill_to_capacity true, design_buffer0, max_candidates_per_design_tick2.
Current begin capacity2/active1/budget1, Phase1 unmet/Phase2 not approved.
B3 is historically terminal; no B2/B4 restart is requested or performed.
Latest explicit user GPU0 grant applies; Y33 stays on GPU1.

## 2. Story Status

| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| W2 confidence is informed by mask-overlap rivals | realized | actual-source Y34 CPU32 | trained semantic necessity unproven |
| Explicit within-mask visual token readout | Y33 training | immutable Y33 admission/launch | frozen full outcome pending |
| Mask visual evidence can query individual words | Y35 mechanism active | CPU32/TRAIN64/native controls | training efficacy/necessity pending |

## 3. Experiment Assessment

| Experiment | mIoU | Contribution |
|---|---|---|
| W2 | historical .5100 | current target comparator, not proof of every submodule |
| B1 | historical .5020 | faithful baseline |
| Y31 | frozen .505514618 | inconclusive against W2/B1, not champion or no-harm proof |
| Y28 | historical rejected | preserve small-group harm of direct visual mean readback |
| Y30 | historical rejected | preserve medium-group harm of ordered trajectories |
| Y34 | not trained | actual existing-path counterexample rejects missing-path realization |

## 4. Gap Analysis

Q1: Do final mask supports evoke task-relevant differences among language tokens,
beyond indirect query conditioning? Q2: Add only the final confidence residual
and retain matched controls, native gradients, timing/memory and frozen outcomes.
Q3: Training can support or refute this specific realization, not establish
global novelty, all-language necessity or benefits from an initialized probe.

## 5. Direction Decision

Y35 follows its prospective preregistration. Equal language means but different
token distributions separate its nonlinear word readout from direct mean pooling.
TRAIN64 uses all words, no GT inference or prediction-based selection.
All64 have positive visual contrast/word variance/wrong-value and wrong-mask
responses; 163/256 have selected-to-oracle gaps. Gap diagnoses room for
confidence selection, not that this path will solve it.
Cached TRAIN8/init4 reach all7 new tensors and preserve other losses/masks.
W2+random-head mean loss+.00222339481 is adverse and retained; short native
learning and physical continuation/resource audits must precede admission.

## 6. Risks And Uncertainty

Initial attention gradients may be tiny, wrong-input controls can also learn,
and correct semantic utility is unproven. Existing upstream attention is present.
Y28/Y30 adverse results are not erased by changing a head. Full evaluation uses
scene-level uncertainty and material subgroup harm, not optimistic monitor peaks.
No external novelty review or complete theoretical proof has been performed.
This is a falsifiable training candidate, not a publication novelty assertion.
Source-copy newline/Windows UTF8 harness failures are preserved; fixes change
tests' encoding comparisons only, not model behavior or eligibility thresholds.
