# Native Inference Counterfactual Sampling

2026-09-30 16:05 Asia/Shanghai, before audit execution.
Use released SPFormer initialization, first8 distinct training scans in
scan_id order, augmentation disabled, original native four-expression chunks.
On each exact same batch run alternating original/GT-zeroed/original/GT-zeroed/
original/GT-zeroed forwards with equal RNG reset. Record selected query,
seed identities, selected-mask probability differences, binary flips.
Compare counterfactual differences with independent repeated-native noise;
do not assume exact sparse equality or tune a numerical tolerance.
Record student-only enabled/disabled warm inference with five paired timings.
This is graph/label isolation and timing, not a trained accuracy evaluation.
Prediction source-path audit complements numerical results. A counterfactual
selection/binary change outside ordinary-repeat behavior requires explanation
before constraints PASS, not a new epsilon. Old exact-equality failures stay.
