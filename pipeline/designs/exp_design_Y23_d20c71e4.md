# Y23 Valid-Token Semantic Support

## 1. Motivation
Goal description: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
Remove nonexistent language alternatives from supervision of query evidence.
This tests training-support correctness, not validation metric calibration.
Parent W2 and all historical rejections remain immutable.

## 2. Current Best Results
W2historicalmIoU.5100/B1.5020/target.515. Y18intermediate.5070 at16/30,
not final. Y23train64 diagnosis: native padding gradients63/63paddedscans,
maskedpaddinggradient/counterfactual0, validtoken/querygradientsnonzero.

## 3. Analysis And Direction
Copy W2 physical package; modify only loss_sem_align to exclude invalid token
support in query-to-token normalization and token-to-query aggregation.
Use exact lang_masks, failclosed for empty/nonprefix language support.
Keep original temperature, weights, target maps and full native inference.
No inference module/parameter additions; novelty and efficacy unproved.

## 4. Experiment Plan
Train from releasedSPFormer for30epochs, seed1999, same W2 config and optimizer.
Native controls: identity/permutation Y22, padding-only intervention Y23,
all-valid equivalence, padding extension, empty/nonprefix rejection,
released-init mechanism backward, tools.train optimizersteps, native resume,
samecheckpoint parent/candidate prediction identity, largest train stress.
No production training until six-gate admission and canonical dispatch.

Post-training frozen evaluation: full ScanRefer validation9508expressions/
141scenes, two independent repeats for Y23 and W2/B1 frozen best checkpoints.
Primary mIoU; also Acc25/Acc50/spmIoU/querycount, latency and peakmemory.
Compare paired per-expression IoU and whole-scene clustered10000bootstrap
draws seed1999; no chunk-based independence or subset tuning.
Subgroups: rawtrain sizecuts1487/3325 frozen by Y16, small/medium/large.
Material-harm rule: paired95% interval wholly below negative empirical
mean-absolute repeat difference for the corresponding comparison/group;
if noise unavailable, few scenes or interval crosses boundary, inconclusive,
not automatic PASS. A single adverse scene flags diagnosis, not rejection.
Target remains.515; no champion promotion from pretraining diagnostics.
Lightweight Pareto: compare fully trained primaryquality, unchangedparameter/
querycount, identicalprotocol singlemodel inference median/p95latency and
peakmemory on the same train8 and frozen largest scene pair after2warmups/
3measurements. Report all cost regressions; no claim that static identity
alone proves fully trained accuracy-cost dominance.

## 5. GPU Allocation Plan
Only server-2GPU0 after live occupation/exclusive lease checks. GPU1Y18
originaltraining untouched. One independenttask/card; no server-6.

## 6. Implementation Checklist
Physical package/config and source audit done; train64/contract/native entry
completed. Inference/backward/resource gates remain pending at creation.
Scoped manual commit only after tested code review; no unrelated dirtyfiles.
Receipt then canonical gated dispatcher, uniquePID/log/stepgrowth verification.

## 7. Risk Assessment
Nuisance effect is small; trained benefit may be absent. Valid-token support
does not repair upstream rawlogitnoise or all alignment semantics.
Native continuation numericaldifferences retained; no full-loader claim.
Misencoded language masks failclosed. Historical failures and Y18 unchanged.
