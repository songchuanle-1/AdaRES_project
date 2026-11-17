# Native diagnostic cycle: O1 and O2

2026-09-26 19:02 Asia/Shanghai. Prepared monitor and goal preflight used once; no training or queue mutation. Native B4-E30 best checkpoint, server-2 GPU0. Inference probes only; no loss backward or admission resource profile.

## O1: Language counterfactual selection
Fixed first/last content-token masking, with unchanged-input repeat control. On 16 validation scenes (441 expressions), repeat mismatch 0, 240 material selection errors. First-token selection flipped in 179/240 errors and 105/201 other expressions; last-token in 169/240 and 100/201. Whole-cohort selected mIoU: baseline 0.49186, first 0.48429, last 0.49538. On 16 separate training scenes (455 expressions), repeat mismatch 0, 107 material errors; first flipped 66/107 errors and 218/348 others, last 66/107 and 193/348. Training selected mIoU fell from 0.83083 to 0.80476/0.81930. The apparent error association is confounded: masking a content token can change referent semantics, and both interventions harm overall training quality. This is not a matched semantic-preserving confidence control. Reject O1; no token-position or fusion scan on validation.

## O2: Global 2D appearance removal
On the fixed first 16 training scenes, repeat mismatch 0. Zeroing all 2D features flipped 101/107 material errors and 321/348 other expressions; selected-mask overlap was 0.344 versus 0.315, respectively. Selected mIoU fell from 0.83083 to 0.28600. Global feature removal is a broad out-of-distribution perturbation without error-specific direction; it does not isolate confidence in a particular query. Reject O2 without spending validation samples on a feature-removal scan.

## Gates and limitations
Both probes were deterministic and ran with peak reserved GPU memory 4.01 GiB (O1 validation), 3.46 GiB (O1 training and O2 training). These are inference-only bounded measurements, not native backward execution, training resources or worst-case memory. Diagnosis and matched-control gates fail; implementation, admission receipt, sync and dispatch are unauthorized. O1 first validation cohort matches the previously fixed 16-scene diagnostic subset, so no numerical intervention may be selected from it. No full literature novelty or theoretical derivation claim. A future independent mechanism needs a frozen paired native control and all six gates before admission.
