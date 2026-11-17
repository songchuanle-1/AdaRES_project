# AdaRES-F2 spatial compactness diagnosis

2026-09-24 14:46 Asia/Shanghai. B4-E30 best checkpoint, first 16 ScanRefer
validation scenes, 441 expressions, read-only native probe on server-2 GPU1.
Ground truth was used only to identify the oracle query after prediction.

The predeclared signal was binary-foreground spatial variance around the
predicted-mask centroid, divided by the scene bounding-box squared diagonal,
on valid superpoints. Smaller values indicate a more compact mask. Among
240 material selection errors (oracle IoU - selected IoU > 0.05), all pairs
were nonempty and only 118 oracles were more compact. Mean normalized
variance: selected 0.00241258; oracle 0.00237760. This is essentially
balanced and does not support the predicted direction.

F2 is rejected at diagnosis: no model implementation, native backward,
resource check, admission, queue entry, or training launch. The bounded
subset does not disprove every geometric mechanism. A new compactness
definition selected after inspecting these validation results would be
feature scanning, prohibited by the goal.

Evidence: `pipeline/analyses/AdaRES-F2_geometry_16scenes_20260924.json`
and `AdaRES/tools/diagnose_adares_f2_geometry.py`. D1 on GPU0 was not
modified. GPU1 remains idle pending an independently justified candidate.
