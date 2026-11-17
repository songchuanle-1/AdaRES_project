# Y34 Diagnosis Preregistration

Registered prospectively during foreground tick a8d0c77d45d142c0a6c46b163b956253.
Stable ID: AdaRES-Y34.
Hypothesis ID: final-mask-overlap-conditioned-query-comparison.

## Causal Hypothesis

The final indicator may lack an explicit comparison of competing queries
conditioned on their predicted spatial support. A readout that compares each
query with other final queries through point-mass-weighted soft mask overlap
could distinguish redundant hypotheses from spatially different competitors.
Existing query self-attention is not presumed absent. Test the actual source
and intervention before claiming a new information path or a task deficit.
This differs from Y33's within-mask token attention; no numeric clone is planned.

## First Discriminating Test

Audit actual W2 final decoder/indicator source and relevant recorded relational
routes. Construct actual-source CPU controls with identical query features and
different mask-overlap graphs, plus jointly permuted queries/masks. Determine
whether the existing final indicator already consumes the same graph, whether
the proposed comparison separates these inputs, and whether it is equivariant.
Persist source hashes, exact controls, repeats and adverse/null results.
Conceptual similarity alone cannot reject this stable candidate.

## Native Diagnosis And Controls

Use the first 64 distinct TRAIN scenes in native dataset order, up to four
descriptions per scene, without model-prediction or validation selection.
Inspect native W2 final masks, queries, indicators, raw point masses and GT
only for diagnosis. Record selected-versus-oracle quality gaps, overlap graph
variation and intended-versus-rolled graph response. GT never enters inference.
Cached identical inputs control upstream sampling/RNG effects.
Freeze enabled, residual-disabled, graph-permuted-only and joint-permutation
controls. Repeat exact cached inputs. No gain threshold or validation tuning.
Expected direction: graph changes affect only the final confidence residual;
joint permutations permute outputs, and all new tensors get finite nonzero
native-loss gradients. Any reverse/null/adverse result must be investigated.

## Prospective Implementation And Limits

Independent physical W2 package, final indicator residual only. Query/key/value
comparison dimension uses parent d_model=256; scale is sqrt(d_model), not a
searched temperature. Use actual native mask probabilities and point masses.
Preserve masks, auxiliary losses, released backbone initialization, 128 queries,
seed 1999, 30 epochs and the single standard argmax. No loss-weight, threshold,
feature-subset, schedule or validation hyperparameter search.
Six gates and the six-signal smell review are pending, not PASS.
GPU0 is authorized but must be live-idle and exclusively leased; Y33/GPU1 is
untouched. Source overlap, utility, novelty and trained Pareto are unknown.

## Frozen Post-Training Comparison

If admitted, compare Y34/W2/B1 on all 9508 frozen validation expressions in
141 scenes, two independent passes per model, first pass primary.
Primary metric mIoU; also Acc25/Acc50, active queries, latency and peak memory.
Whole-scene clustered bootstrap: 10000 draws, seed 1999. Raw target point-count
cuts 1487/3325 come from TRAIN, not validation tuning. Material subgroup harm:
paired interval upper bound below zero against either W2 or B1.
Intervals crossing zero are inconclusive, not equivalence or absence of harm.
Measure repeated-evaluation noise and trained paired cost; invent no noise floor
or pre-training efficacy/Pareto claim. Target remains 0.515.
