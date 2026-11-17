# AdaRES-Y40 Prospective Diagnosis

- Frozen before source-family review and candidate outcomes, 2026-10-11.
- Hypothesis ID: final-mask-word-support-alignment-confidence.
- Hypothesis: a final confidence readout with explicit valid-word to mask-supported feature alignment may distinguish expression/support pairings lost by a mean-pooled mask feature readout. Upstream decoder states may already encode this information; neither a native deficit nor benefit is established.
- Matched control: physical W2 indicator and the exact existing realization if source review identifies one. Fix final queries, masks, mask mass, support-feature means, all dimensions, parameters and valid-token padding. Contrast word-support pairings with equal token means; repeat and joint permutation controls. No validation-derived cutoffs or parameter sweep.
- First test: inspect actual final-indicator input contracts and run a CPU, prediction-blind intervention on any existing readout that consumes valid token/support alignment. Existing path sensitivity plus identical proposed statistics would disconfirm an uncovered realization; conceptual overlap alone does not.
- Native continuation if uncovered: freeze implementation and native TRAIN-only sampling, measure correct versus rolled-word/wrong-support controls, all new tensor gradients, native backward, time and memory. Keep masks, native 128 queries, one argmax, released SPFormer, fixed30/seed1999 and original losses unchanged.
- Admission and training remain forbidden until all six executable gates pass. Synthetic sensitivity is not efficacy or necessity. No novelty claim.
- Preserve original Y39 GPU1 training. Handle Y37 done_complete using its frozen full9508/141, two-pass FIRST-primary comparison against W2/B1, scene bootstrap10000/seed1999, TRAIN cutpoints1487/3325, subgroup harm, repeat noise and trained cost plan.
