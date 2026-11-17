# Specificity Sampling Detail

Frozen 2026-09-30 15:52 Asia/Shanghai before specificity execution.
Source: AdaRES-Y15_train64_diagnosis_20260930.json, not new model outputs.
First eight material-bearing actual scan_ids in source order:
scene0000_00, scene0002_00, scene0006_00, scene0008_00,
scene0010_00, scene0012_00, scene0014_00, scene0016_00.

For each scan choose its first previously material identity
(scan_id, object_ids, ann_id), plus the first three source identities with
distinct object_ids from that identity and from already chosen peers.
Use native augmented dataset on these explicitly frozen expression groups.
This diagnostic single-scene four-expression cohort differs from the native
batch2 resource/backward tests; it is not a substitute for their evidence.
Correct-versus-wrong compares B4 oracle masks for the respective true objects
on the SAME augmented scene. Wrong control must be a different object and
have the same superpoint coordinate system; never compare masks across scans.
Only current material events enter teacher-specific gradient summaries.
If none occur, teacher-specificity is unavailable rather than PASS.
Original cohort diagnosis and prior assertions remain unchanged.
