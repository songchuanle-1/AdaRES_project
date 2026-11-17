# AdaRES-F1 mask-uncertainty diagnosis

2026-09-24 14:43 Asia/Shanghai. Read-only B4-E30 best-checkpoint diagnostic on
server-2 GPU1, fixed first 16 ScanRefer validation scenes (441 expressions).
No training, deployment of model weights, or change to D1.

The proposed F1 signal was mean binary entropy of each predicted mask over
valid superpoints. The predeclared diagnostic asked whether the missed oracle
had *lower* entropy than the selected query in B4's material selection errors.
Ground truth identified oracle queries for analysis only.

Of 240 errors with oracle IoU minus selected IoU > 0.05, just 86 had a
lower-entropy oracle. Mean entropy was 0.00446 for selected queries and
0.00632 for oracles. Choosing the minimum-entropy query without labels
gave subset mIoU 0.0262 versus B4's original 0.4919. The evidence
contradicts the stated selection mechanism in this bounded subset.

F1 is **rejected at diagnosis**. No F1 model, config, admission receipt or
training job was created. A different entropy definition chosen by scanning
validation results would violate the project's no-feature-tuning rule.
The 16-scene result cannot prove that uncertainty never helps other tasks.

Raw evidence: `pipeline/analyses/AdaRES-F1_mask_uncertainty_16scenes_20260924.json`.
Native probe: `AdaRES/tools/diagnose_adares_f1_mask_uncertainty.py`.
