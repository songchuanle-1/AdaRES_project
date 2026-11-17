# Fixed native diagnosis: V1 and V2

2026-09-27 02:00 Asia/Shanghai. Prepared monitor/goal preflight used once, not rerun. Key SSH to server-2, GPU0, B4-E30 best checkpoint, first 64 ScanRefer training scenes (1660 expressions). Read-only inference. No validation-based selection or numerical scan.

## V1: Local feature-boundary alignment

For each predicted binary mask, compare local projected 3D+2D feature distance across its spatial boundary against non-crossing edges. Geometry-only distance on the same nearest-neighbor graph is a matched control. Fixed maximum 512 superpoints per scene and four nearest neighbors bound computation; neither was selected on validation. Among 337 material errors with both boundaries, missed oracle feature alignment was higher in 113 versus geometry-control ordering 117. Feature-only mIoU 0.40379, geometry-only 0.30367, B4 0.82256. Repeat mismatch 0. Reject: the proposed direction fails and does not exceed its matched ordering control. Peak inference reservation 3.58 GiB is not training resource evidence.

## V2: Query-to-mask feature consistency

The final query embedding's cosine agreement with its predicted mask's pooled scene features is compared with the same query and outside-mask pooled features. Of 388 material errors, missed oracle inside agreement was higher in 145; outside control in 123. Inside-only mIoU 0.05337, outside-only 0.11130, B4 0.82256. One repeated score mismatch at 1e-4 despite identical selected IoU; further reproducibility work would be necessary for any successor. Reject the direction and standalone selector. Peak inference reservation 3.56 GiB is not training resource evidence.

## Gates

Both routes fail diagnosis before any candidate-specific backward, training memory/latency, independent model/config, admission receipt, sync, queue mutation or dispatch. No global exhaustion claim. Raw paired results and scripts are hashed in research_work.json. The fixed training subset is diagnostic, not a population metric or novelty proof.
