# Fixed native screens: sampler rank and mask area

2026-09-26 15:01 Asia/Shanghai. B4-E30 best checkpoint, first 16 ScanRefer validation scenes, 441 expressions, 240 material selection errors (oracle minus selected IoU > 0.05). Read-only native server-2 GPU1 inference. Raw paired rows: `AdaRES-K1-K2_native_16scenes_20260926.json`; script: `AdaRES/tools/diagnose_adares_k1_sampler_rank.py`. These subset observations do not estimate full-validation effects.

| Route | Predeclared mechanism | Paired evidence and controls | Decision |
|---|---|---|---|
| K1 sampler-rank shortcut | B4 confidence overuses the sampler's relevance ordering, systematically suppressing later high-IoU queries. | Oracle later than selected in 141/240 errors; oracle sampler score higher in 99/240. Sampler-only top-1 mIoU 0.34587, reversed-order control 0.03610, B4 indicator 0.49186. Sampler-only fixes 50 errors and worsens 43. | Reject this shortcut mechanism. The index difference alone is not causal evidence; no rank-weight scan or training admission. |
| K2 mask-area shortcut | B4 confidence favors a systematically wrong mask scale over good target masks. | Oracle area larger in 118/240 errors, close to a split. Largest-mask top-1 mIoU 0.27271, smallest-mask 0.00000, versus B4 0.49186; largest-mask fixes 49 errors and worsens 48. | Reject this scale-bias mechanism. No area threshold or fusion scan. |

The sampler uses descending top-k relevance order, so raw index has a structural interpretation, but the selected-vs-oracle index asymmetry alone does not demonstrate that the head uses rank. K2 uses point-mask area without target information for the controls. Neither passes the diagnosis hard gate; no candidate-specific native backward, resource or admission receipt is claimed. G1 training remains a separate test.
