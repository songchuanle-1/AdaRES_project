# Y37 Fixed Realization And Sampling

Prospective, before CPU/native outcomes. Preregistration:
pipeline/analyses/Y37_diagnosis_prereg_e636ce91.md.

Use point-mass-weighted soft foreground/background distributions over valid
native superpoint centroids. For two independent draws with replacement,
compute exp(-k * squared_distance / scene_weighted_coordinate_variance),
k=1,2,3 fixed by the three coordinate dimensions. Summarize FG/FG, BG/BG,
FG/BG expectations. Shared Linear3->256/GELU/Linear256->256 encoder; concatenate
final normalized query, encoded FG-FG minus BG-BG and encoded FG-BG;
Linear768->256/GELU/Linear256->2 residual to final indicator only.
No learned scale, radius, temperature, loss weight, feature selection or search.
The three chosen kernel orders are a conservative fixed realization, not a
uniquely derived optimum; no realization selection by outcomes is permitted.
Parent initialization uses fork_rng so shared initialized tensors are unchanged.

CPU:32 seeds, repeat/permutation/query permutation/translation/orthogonal
rotation/uniform-scale/padding/mask-detach/wrong-mask/wrong-coordinate controls.
Equal-moment fixture xA=(-3,-3,-1,-1,1,1,3,3), xB=(-4,-2,0,0,0,0,2,4);
both exact mean0/secondmoment5, uniform masks/mass, constant identical visual
features. Test actual W2, Y31, Y36 and Y37 extracted classes, not a mock.
Tolerance1e-5 for floating invariances, exact repeat/detach/padding unchanged.
Readout effect direction is distribution separation, not initialized loss gain.

Native diagnosis:first64 distinct TRAIN scenes in dataset order, at most4
descriptions per scene inherited dataset loader; fixed W2 best checkpoint.
Measure oracle selection gap only for diagnosis, kernel contrast FG-FG versus
BG-BG, wrong-mask/coordinate response, repeat/permutation. Ground truth never
enters the inference path. Preserve all rows, not only responding scenes.
Native cached controls:first8TRAIN scenes plus first4 released-init scenes,
disabled/enabled/repeat/detached-mask/wrong-mask/wrong-coordinate.
All8 new tensors must receive finite nonzero loss gradients.
Use native tools.train two steps and exact model/AdamW/PolyLR restoration;
report numerical continuation differences, do not demand or claim global
bitwise determinism.

Resources:first8 TRAIN scenes plus fixed maximum pair728/6780; two warmups,
three timed readings; count params, full native loss/backward/AdamW peak,
native128/oneargmax inference. New params<1% W2, stress reserved<24GiB/noOOM.
Quadratic superpoint pair time/memory is an explicit risk. Cost is measured,
not called trained Pareto or speedup.

Post-training:full validation9508/141, candidate/W2/B1 twice/first primary;
all mIoU/Acc25/Acc50/querycount/latency/peakmemory, scene10000 bootstrap seed1999,
TRAIN point cuts1487/3325, subgroup CIupper<0 against either is material harm;
CIcross0 inconclusive, independent repeat noise, trained paired cost.
Promotion requires target .515 and frozen harm/Pareto review. No efficacy,
semantic necessity, upstream blindness, human judgment or novelty assertion.
