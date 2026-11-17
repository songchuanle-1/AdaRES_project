# Y17 Native Admission Review

2026-09-30 Asia/Shanghai. Stable same-target-description-mask-consistency.
This review permits a falsifiable 30-epoch training test, NOT an efficacy claim.

## Diagnosis And Controls

Train-only same-target description prediction disagreement .910843 overlap,
repeat 1, wrong target .093917. No validation-directed hyperparameter choice.
Physical implementation retains native losses and adds scene-balanced,
point-mass weighted Bernoulli JS with preregistered unit coefficient.
Same forward tensors: pair-off exact zero; total exactly native+JS; wrong
target diagnostic gradients differ in all four batches. Same/wrong combined
gradient cosine .859-.950 shows substantial generic smoothing at initialization.
Earlier confidence-only native-gradient opposition remains recorded; fresh
combined gradients do not erase it. No fixed pre-training effect-size test.

## Native Path And Resources

Physical released-backbone four-batch loss/backward finite; all 14 tested
confidence/input-projection tensors have nonzero finite JS gradients and native
optimizer updates. No newly introduced trainable tensors. Inherited absent
native gradients are not described as newly connected paths.
Native main with the proposed entry, full datasets and four-worker loader
executed two optimizer steps; val dataset unchanged, no val metrics examined.
Actual encoded population: 15,129 pairs, 30,258 paired expressions,
7,649 of 9,372 chunks; all 36,665 train annotations preserved.
510 exhaustive sampler tests; no-pair graph zero; wrong-object/self/cross-scene/
duplicate positives fail closed. Identical encoded text excluded by collate.

Frozen large train batch 1,055,503 points/7,618 superpoints fits 14.082 GiB
reserved; measured steps .911-.923 sec. No OOM. Large-scene JS is very small;
resource feasibility is not a mechanism benefit. Parameters 164,629,637,
128 queries, one native argmax-selected query, no inference teacher.
Train8 repeat/same-weight W2/GT-counterfactual tests all binary, seed and
selection flips zero; probability noise approximately 1e-6. Timing has both
models resident, so recorded process peak is NOT a standalone inference peak.
Gorilla restores model/AdamW/PolyLR exactly before continuation except known
nonpersistent Roberta buffers. Next-step weight difference 5.87e-5 preserved,
no bitwise or whole-loader RNG-resume claim.

## Deployment Safety

One reviewed native training dataset hook, old W2/Y15 fallback identity tested.
No production cross-variant imports; YAML equal to frozen W2.
Project sync exception limited to tools/train.py from parent 2bb1c107...
with exact old/new hashes, atomic replacement and fail-closed stale marker.
Isolated exact remote-body tests cover foreign-byte rejection, update,
idempotence, stale-parent rejection and unchanged source marker.
No global gate/policy/scheduler changes; running Y15 process untouched.

## Frozen Evaluation And Risks

Server-2 GPU1 only, one task/card; exactly30 epochs/seed1999, released SPFormer
backbone. Official full ScanRefer val mIoU versus W2 .5100/B1 .5020, target .515,
Acc25/50, spmIoU, queries/latency/standalone memory. Use scene clusters and
train-size cuts1487/3325, independent-repeat noise and material-harm/Pareto
rules already frozen in Y17 paired-training design/Y15 referenced plan.
Noise floor not yet quantified; crossing intervals => inconclusive.

Regrouping changes chunk context; causal efficacy needs the EXACT regrouped
pair-off post-training control, not historical W2 alone. A second control run
is not independently admitted capacity filler. Consistent wrong predictions,
tiny JS, singleton exclusion and mixture/argmax mismatch remain possible.
Literature/novelty is not established by these engineering tests.
