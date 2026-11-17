# Motivation Review - AdaRES_project

2026-09-27 02:00 Asia/Shanghai

## 1. Goal
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515. B1 0.5020, champion B4-E30 0.5094, gap 0.0056. Phase 1 unmet. Goal constraints and bans checked: server-2 GPU0/1 only, single-GPU tasks, released backbone, project decoder, fixed B2/B4 continuation, required metrics/resources, no validation scanning.

## 2. Story Status
B4 query competition is positive over B1 but below target. Prior topology, compactness, feature perturbation, attention, language, and direct cross-modal evidence routes were rejected on bounded diagnoses. V1 asks whether a predicted mask's local boundary follows a feature discontinuity; V2 asks whether its query representation agrees with features under its mask.

## 3. Experiment Assessment
Fixed train64 B4 selected mIoU 0.82256. V1 feature-only 0.40379 and oracle-higher ordering 113/337, geometry control 117/337. V2 inside-only 0.05337 and oracle-higher 145/388, outside control 123/388. These are native inference diagnoses, not trained experiments.

## 4. Gap Analysis
Neither proposed signal robustly identifies missed-oracle masks. V2 has one repeated-score mismatch. Active queries, full-run latency and training peak memory remain unmeasured for these unadmitted routes.

## 5. Design Rationale
V1 tests local feature transitions independent of prior connectedness and spatial compactness. V2 tests query-mask consistency independent of token-mask and cross-modal matching. Their matched controls disconfirm the stated mechanisms; fusion or parameter scans would not repair the evidence.

## 6. Next Batch
No admission. Resume only with an independently justified inference-available, metric-aligned mechanism and predeclared native paired controls. Then test backward, resources, independence and all goal constraints before receipt, sync and safe dispatch.
