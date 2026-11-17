# Motivation Review - AdaRES_project
2026-10-03 21:13+08; token ed4dc9656e8f4556a57c17cb369279c6.

## 1. Goal
Canonical pipeline/goal.json: metric mIoU, target0.515, dataset ScanRefer,
baseline AdaRES-B1. Description: "AdaRES: improve a faithful standard-indicator
3D referring expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."
Constraints: server-2 only, single-GPU/max one task per card; no successor while
B3 runs, at most two independent successors after its terminal state; B2/B4
fixed30 with optimizer/scheduler-preserving extension; released SPFormer
backbone/project decoder; report mIoU/Acc25/Acc50/active queries/latency/peakmemory;
no validation tuning of thresholds/temperatures/weights/features/epochs.
Bans: no server6, no released b005/CFNet as B0, no margin/occupancy-weight scan.
Policy: research_on_capacity_gap/fill_to_capacity true, steady_ticks_before_design1,
design_buffer0, max_candidates_per_design_tick2. This tick actual budget1.
Newer user override: GPU0 reserved, current whitelist GPU1 only. B3 long terminal,
B2/B4 extension untouched. W2 historical0.5100/gap0.0050; Phase1 not reached.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Metric-aligned point competition | W2 positive historical comparison, not target | W2 .5100/B1 .5020 | Generalization and causal necessity |
| Point-measure loss Y18 | Inconclusive after full frozen protocol | .507606 vs W2 .510010, clustered CI crosses0 | Not a champion |
| Padding-normalization Y23 | Mechanism present, trained efficacy inconclusive | .507472 vs W2 .510105, CI[-.010862,.005472] | Not a champion |
| Indicator target shortcut Y24 | Actual realization rejected | 32 native CPU controls: target detach loss/grad0 | No executable new path |
| Mask-feedback lockout Y25 | Diagnosis/control activity only | TRAIN64 target mass excluded .087-.156; actual attention32 support/gradient restored | Native loss path, cost, independence, efficacy |

## 3. Experiment Assessment
| Experiment | mIoU | Relation To Goal | Contribution |
|---|---|---|---|
| W2 | Historical .5100 | Query confidence from point-mass competition | Preserve comparator |
| Y15 | .230319 | Additional mask selection | Realized selection collapse, rejected |
| Y17 | .504157 | Same-target mask consistency | Trained comparison inconclusive |
| Y18 | .507606 | Exact point-measure mask supervision | Trained comparison inconclusive |
| Y23 | .507472 | Valid token semantic normalization | Full frozen comparisons complete, no reliable benefit/harm |
| Y24 | Not trained | Target-only stop-gradient | Native binary target is already nondifferentiable |
| Y25 | Not trained | Confidence evidence depends on trainable mask refinement | Distinct untested feedback path |

## 4. Gap Analysis
No trained successor yet clears0.515. Confidence cannot rank evidence it never
receives: a previous detached hard mask can exclude real target points before
later query refinement. TRAIN64 proves exclusion, not that it is a bottleneck.
Q2 rejected support-based oracle sorting (70/388 errors); Y25 does NOT reopen
that selector or scan its threshold. It changes TRAIN mask-to-attention feedback
and preserves eval hard masks, with native/full/detach controls. Positive target
labels are diagnostic readouts only, never input to the intervention.

## 5. Design Rationale
Y25 tests whether masks can receive native downstream attention gradients while
remaining a query-specific prior over all valid point evidence. Fixed
logsigmoid(logits), no coefficient/temperature/search; detach same forward
isolates feedback gradient, full-valid isolates unconditional access.
Costs and native graph must pass before admission. A positive trained result
would support a refinement-evidence route to confidence; a negative result
would disconfirm this realization, not justify another numeric setting.
No literature search or novelty claim; training/eval mismatch is a real risk.

## 6. Next Batch
See pipeline/designs/exp_design_Y25_ed4dc965.md. Prototype native first8 trained
and first4 released-init tests are next; physical independence, native two-step,
resume, peak memory and inference identity remain pending. No queue entry yet.
