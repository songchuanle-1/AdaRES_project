# Y20 Cached-Input Repeat Audit

Frozen2026-10-02 before this follow-up output; original64scene evidence retained.

- Original train64 native/repeat final indicator logits differ up to8.66407 despite almost equal loss/cosines; query/FPS permutation is possible but unproved. Do not claim exact native end-to-end forward equivalence.
- Follow-up samples first8 distinct native training scans in original order, same W2 frozen checkpoint/seed1999; no validation.
- Hook the final QueryCompetitionIndicator query/masks/batch_mask/point_mass and clone these actual tensors. Execute native/native-repeat/detached on exactly the same tensors and same module state.
- Readouts: exact maximum logit differences, indicator gradient to masks absent under detach, head gradients nonzero, cosine sign/mean difference across all64 original native repeats, mean absolute indicator/mask loss repeat differences.
- This isolates backward-routing identity from native upstream ordering/noise; cannot erase original discrepancies or prove full-model determinism. Native64positive scene-mean already falsifies the preregistered global opposing-direction premise; do not select the21negative scenes for admission.
