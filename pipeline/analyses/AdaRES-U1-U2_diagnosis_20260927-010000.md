# Fixed native diagnosis: U1 and U2

2026-09-27 01:00 Asia/Shanghai. Prepared monitor and goal checks were not rerun. Server-2 GPU0, key SSH, B4-E30 best checkpoint, fixed first 64 ScanRefer training scenes (1660 expressions). Read-only inference, no validation tuning. Repeated forwards matched in both probes.

## U1: Masked cross-modal agreement

The proposed signal compares pooled projected 3D and 2D features inside each predicted mask. Outside-mask and cyclically shifted 2D features are controls. Of 388 material B4 selection errors, the missed oracle had greater inside agreement in 148, outside agreement in 207, shifted agreement in 165. Agreement-only top-1 mIoU was 0.09220, versus B4 0.82256, outside-only 0.38530 and shifted-only 0.03072. This is neither discriminative nor superior to its controls. Inference peak reservation 3.60 GiB does not qualify training resources.

## U2: Token-resolved mask-language compatibility

The proposed signal takes the maximum valid-token cosine match to the predicted mask's foreground-minus-background scene representation. A scene-shifted language match is the control. Of 388 material errors, oracle match was greater in 179, versus 182 with shifted language. Token-only mIoU was 0.00814 versus B4 0.82256 and shifted-only 0.01109. This does not establish expression-specific metric evidence. Inference peak reservation 3.56 GiB does not qualify training resources.

## Gates

Reject U1 and U2 on diagnosis/control evidence. No candidate-specific native backward, training resource test, independent model/config, admission receipt, training sync, dispatch, or queue mutation. This is bounded exhaustion of these two routes, not a claim that all hypotheses are exhausted. Reconsider only on a prospectively specified inference-available mechanism with matched native controls and all six gates.
