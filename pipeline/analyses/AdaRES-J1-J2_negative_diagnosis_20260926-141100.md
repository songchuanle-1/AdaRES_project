# Independent fixed-subset screens after H5

2026-09-26 14:11 Asia/Shanghai. B4-E30 best checkpoint, fixed first 16 ScanRefer validation scenes, 441 expressions. Read-only native GPU1 inference, no training or selection-policy changes. Raw paired rows: `AdaRES-J1-J2_native_16scenes_20260926.json`; script: `AdaRES/tools/diagnose_adares_j1_mask_language_evidence.py`.

| Route | Predeclared mechanism | Observation on 240 material selection errors | Disposition |
|---|---|---|---|
| J1 | Target mask foreground-minus-background scene features better match the projected language summary than a wrongly selected mask | Oracle contrast higher in 123/240. Contrast-only top-1 subset mIoU 0.01315 versus B4 0.49186; a scene-only control reached 0.00119. | Reject this pooled-language contrast route; no validation feature or weight scan. |
| J2 | Overlap-crowded oracle masks are suppressed by B4's rival context | Oracle crowding higher in only 78/240; mean oracle crowding 6.487 versus selected 7.936. | Reject duplicate-mask suppression mechanism as stated. |

The J1 pooled text summary is a diagnostic proxy, not a trained cross-attention control. A poor standalone ranker does not refute every learned mask-language mechanism, but the paired error contrast fails this route's proposed direction. J2 uses B4's own soft overlap definition without a new threshold. Both are bounded observational tests and not full-validation effects. Neither passes the diagnosis gate; controls, native backward, candidate-specific resources, independence and admission receipt are not claimed. GPU1 remains vacant rather than filling it with a placeholder.
