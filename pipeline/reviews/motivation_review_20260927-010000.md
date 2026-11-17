# Motivation Review - AdaRES_project

2026-09-27 01:00 Asia/Shanghai

## 1. Goal
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515. B1 0.5020; B4-E30 0.5094; gap 0.0056. Goal constraints and bans checked: server-2 GPU0/1 only, single-GPU tasks, released backbone and project decoder, fixed B2/B4 continuation, required metrics and resources, no validation scans.

## 2. Story Status
Query competition has a positive B4 result but misses the target. Existing overlap, mask geometry, modality perturbation and pooled-language directions were negative. U1 tests direct cross-modal agreement; U2 tests token-resolved mask-language evidence, not pooled-language contrast.

## 3. Experiment Assessment
B4-E30 remains champion 0.5094. U1/U2 are fixed train64 native inference diagnostics, not trained experiments or validation estimates. U1 oracle ordering 148/388 versus outside 207/388; U2 179/388 versus shifted-language 182/388.

## 4. Gap Analysis
Neither signal distinguishes missed oracles from wrong selections reliably. Active-query, latency and training peak-memory metrics remain unmeasured for unadmitted variants.

## 5. Design Rationale
Independent cross-modal and token-specific routes tested with predeclared controls. Both failed before backward and resource checks; extending them by numerical fusion or token subset scans would not repair the causal evidence.

## 6. Next Batch
No admission. Seek a new inference-valid mechanism with native paired diagnosis/control, backward, resource and independence gates; do not rename U1/U2 or tune on validation.
