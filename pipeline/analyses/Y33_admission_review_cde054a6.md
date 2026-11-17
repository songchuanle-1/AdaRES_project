# Y33 Gate Review

Owned tick cde054a6b0254e37a0c0d71febbcc497, 2026-10-08.
Autonomous review, not human approval, efficacy or novelty certification.

## Diagnosis

Prospective prereg4b008393, actual-class CPU08a079ba and native TRAIN64ed6513be.
Equal FG/BG means do not fix the actual attention output (delta0.052794486).
TRAIN64/256 has163 expressions with candidate-choice point-IoU gaps; all64
token variance and wrong-token/wrong-mask responses nonzero. This is a
task-relevant, falsifiable direct information-path hypothesis. Upstream visual
attention already exists; absence of direct readout is not global blindness.
No minimum task-effect-size criterion or validation selection was applied.

## Controls

Native cached TRAIN8/W2 and released-init4 0f8dd53c: disabled/repeat/
mask-detach/wrong-mask/wrong-token controls. All seven new tensor gradients
finite/nonzero, minimum7.325722e-5; mask-gradient intervention minimum3.813268e-7.
Mask-detach forward and repeated indicators exactly identical, all
nonindicator loss deltas0, joint permutation max2.682210e-7. Wrong-mask/token
response minima0.000448495/0.002449721.

Original adverse W2-trained native mean loss delta+0.000569448 and released-init
delta-0.068547368 retained. Fixed native AdamW/lr0.0001/weight_decay0.0005/
grad_clip0.1 two-step investigation fd7e96e1 changed only temporary new7
tensors: trained enabled/rolled/wrong-token changes
-0.001673691/-0.001676209/-0.001592912; released-init
-0.031022072/-0.031020522/-0.030663371. All controls learn, some individual
scene losses rise. This investigates random-residual learnability, not useful
token necessity or elimination of the original adverse result. No parameter,
setting, training checkpoint or deployment initialization selected from it.

## Native Backward

53751573: real tools.train two AdamW steps, all seven tensors nonzero/updated,
407 decoder gradient tensors finite. Exact saved model/optimizer/PolyLR restore
before continuation. Next-step loss identical but model max difference1.525879e-5
and optimizer6.073037e-9: do not claim bitwise trajectory equivalence.
Cached four-scene inference: disabled physical W2 masks/indicators/sampling
identical; enabled final readout leaves masks unchanged; native128/one selection.

## Resources

92ad3af9: identical shared parent initialization verified, serial native
train2warmups/3readings, TRAIN8 plus frozen stress728/6780 inference.
Parameters164958087 versus164629637, +328450 (about0.20%, below existing1%
added-parameter operational screen). Stress median0.904157039 versus0.899823178s;
reserved13.677734375 versus13.40234375GiB, below24GiB. Inference median
0.093264481 versus0.089862733s, descriptive p95
0.215372375 versus0.212133639s, reserved1.7578125GiB both.
Small fixed-order serial samples do not prove significance or speedup.
Resources fit the existing lightweight execution screen; trained Pareto
quality/cost qualification remains pending, not certified here.

## Independence And Constraints

66f91ab9 guarded seven-file source bundle: physical W2 copy, no cross-variant
imports, unchanged nondecoder AST/config. Actual new class CPU/source controls
pass. Only final indicator receives token-attention residual; native masks,
128 queries, inference argmax, auxiliary objectives and loss weights unchanged.
Goal53da8627 untouched; only server-2 GPU1 authorized, GPU0/server6 untouched.
Fixed30/seed1999/released SPFormer and frozen full9508/141/three models twice/
whole-scene10000/TRAINcuts1487/3325/material-harm/trained-cost protocol bound
in exp_design_Y33_cde054a6.md. All six smell signals reviewed there; this is
new aggregation rather than a numeric-setting scan. Y28/Y30 rejected
realizations and Y32 source counterevidence remain unchanged.

## Pre-Admission Disposition

Six executed gates support a falsifiable training admission, NOT expected
trained gain. Before receipt/queue write, scoped candidate files must be
committed and exact git/local/deployed LF bytes verified with unchanged AST;
fresh CPU source controls must run on aligned bytes. Canonical additive-marker
sync and native launcher are explicit project policy, not an origin-push claim.
Unique training log must show increasing native steps; a PID alone is pending.
Production training, admission receipt and postflight are not claimed by this
pre-admission review.
