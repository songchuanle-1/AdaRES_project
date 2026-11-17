# Y30 Admission Review

Owned tick86766fcd883d46c1a9d60563f4213c2e, 2026-10-06, after all actual
precision-repaired probes and before source commit/admission/production launch.
This review authorizes a falsifiable fixed-protocol experiment, not promotion.

## Diagnosis And Motivation

Stable Y29 was rejected: source premise false and32 fixed loss/gradient cases
exactly reproduce historical B0 ranking (0/0). No renamed margin/mass variant.
Stable Y30 has a different mechanism: seven aligned same-query decoder states
and point-mass mask transitions feed a learned GRU confidence residual.
This explicit ordered input path is absent from W2's terminal confidence head;
implicit history in upstream states already exists. TRAIN64/256 retains163
oracle gaps/all163 distinct trajectories, scene mean gap.15414838906144723.
This diagnoses a testable information path, not trained benefit or necessity.

## Six Gates

| Gate | Actual Evidence | Executability Judgment |
|---|---|---|
| diagnosis | y30_trajectory_diagnosis_86766fcd883d46c1a9d60563f4213c2e.json, prediction-blind TRAIN64, preregistered | PASS information-path diagnosis, efficacy unknown |
| controls | y30_native_controls_fp64_86766fcd883d46c1a9d60563f4213c2e.json, TRAIN8/init4 | PASS activity separated from disabled/rolled/reverse/terminal; repeat/permutation/nonindicator0, mask-detach forward0/gradient effect3.1269434e-7 |
| native_backward | y30_physical_native_fp64_86766fcd883d46c1a9d60563f4213c2e.json | PASS native tools.train2steps,8 new tensors all finite/nonzero/updated,408 finite decoder gradients; exact model/AdamW/PolyLR restore |
| resources | y30_resources_fp64_86766fcd883d46c1a9d60563f4213c2e.json | PASS measured24GB feasibility and<1% new parameters, NOT trained Pareto |
| independence | source_bundle_fp64 and structure_cpu_fp64, actual byte/AST/config identity | PASS independent6-file package, exact parent YAML, no cross-variant imports, CPU32 |
| constraints | frozen exp_design_Y30_86766fcd.md, motivation review, precision repair declaration | PASS prospective six-gate policy, goal frozen, native training/inference preserved, GPU1 only |

All evidence hashes are bound in the stable ledger and admission receipt.
Required gates judge an executable/falsifiable graph, not expected mIoU.
New8 tensors397314 parameters, total165026951 versus164629637 W2.
GRU hidden width256/native7 states,128 active queries/one argmax selection.
Same native losses/backbone/sampling/aux outputs; final indicator residual only.
Shared full-model initial tensors remain bitwise equal with CPU fork_rng.

## Faults And Numerical Reconciliation

Original physical optimizer audit used an incorrect qualified-name filter;
the measurement-only v2 fixed the filter, not the model/gate.
Original native control v1/v3/v4 failures are retained. V4 established
superpoint permutation error1.4424324035644531e-5>unchanged1e-5, while
nonindicator deltas were0. An earlier working attribution to a nonindicator
failure was corrected, not retroactively certified.
The separate numerical diagnosis376f2dae isolated FP32 geometric-reduction
noise. Same formula in FP64 yielded cast descriptors/readout permutation0
for all12 fixed cases. Original source32332300 preserved; only the new mask
statistics use FP64 before cast back to query dtype. No coefficient, sample,
threshold, formula or learned initialization changed. Repaired source
d5b6ecc0 was byte-verified and ALL physical gates rerun, not inherited.

## Adverse Results And Costs

Repaired cached initial task-loss deltas versus disabled: trained mean
-.0004016086459159851, released-init mean+.1383742094039917.
Wrong-history/reverse/terminal outputs differ, but several controls have
smaller loss or learn similarly. Two temporary new-path-only optimizer steps
reduce correct mean loss-.002306796610355377/-.10930705070495605 for
trained/init; reverse and terminal changes also descend. This investigates
learnability of initialization, not correct-order necessity or population
benefit. All per-scene/adverse values retained; diagnostic weights not deployed.

Native resume before continuation is exact; next loss difference0,
model max1.52587890625e-5/optimizer4.362846084404737e-9 retained.
No whole-loader/RNG or bitwise trajectory equivalence.
FixedTRAIN8+largestpair728/6780, serialbatch1 inference,2warmups/3readings:
stress medianW2/Y30 .904825868/.913789744 seconds;
stress reserved13.40234375/13.7578125GiB.
Inference median .091597872/.090580111 seconds, descriptivep95
.218310121/.218485553, reserved1.7578125/1.759765625GiB.
Measured small cost increases, mixed timing, no significance/tail/Pareto claim.

## Freeze And Smell Check

Fixed30epochs/seed1999/released SPFormer/unchanged AdamW-PolyLR/argmax.
Full9508expressions/141scenes, independent two passes each Y30/W2/B1;
whole-scene bootstrap10000seed1999/TRAINcuts1487/3325/material subgroup harm
rule/serial trained cost remain as frozen in the original design.
No validation tuning, copied checkpoint warm start, teacher, pool/selector
change, foreign project, GPU0, scheduler or background restoration.

New learned ordered cross-stage interactions and two physical mask-transition
statistics constitute a nontrivial information aggregation path, not a
norm-only/gate-only/scale-only change or fixed score averaging. Numeric values
derive from the parent's native hidden width/stages, not a scan. The precision
repair fixes mathematical permutation fidelity, not a selected scientific
variant. Literature novelty, temporal necessity, trained efficacy, subgroup
harm and training-seed uncertainty remain UNKNOWN. Preserve Y28 rejection.

## Decision

Commit only the seven byte-tested Y30 package/config files. Bind that exact
HEAD, current goal hash and this review/decision into six-gate admission.
Canonical dispatch only if authorizedGPU1 is live idle and project capacity
still empty. Native increasing steps are required; claiming/pending must be
reconciled without duplicate launches. No efficacy promotion at admission.
