================================================================================
  Motivation Review - AdaRES_project
  2026-09-21 14:50:43
================================================================================

## 1. Goal
  - Target: ScanRefer mIoU >= 0.515.
  - Champion: AdaRES-B2, mIoU 0.505; gap 0.010.
  - Baseline: AdaRES-B1, mIoU 0.502.
  - Constraints: server-2 GPUs 0 and 1 only; exactly one GPU per new run;
    no server-6; no ScanRefer validation tuning of scalar hyperparameters.

## 2. Evidence
  - B0 showed that strong candidate masks can coexist with poor top-1 query
    ranking: oracle mIoU 0.847 versus final 0.333.
  - B1 established the faithful learned-indicator baseline at mIoU 0.502.
  - B2 improved the same learned indicator with matched-query IoU supervision
    to mIoU 0.505, showing that metric-aligned targets help but are insufficient.
  - B3 added direct mask-conditioned scene evidence and closed at mIoU 0.5014.
    A repeat or scalar variation of that mechanism is not admissible.

## 3. Distinct Remaining Gaps
  - B1-B3 score every query without explicitly comparing it to the competing
    query set. This leaves duplicate or overlapping candidates uncalibrated
    relative to one another.
  - B1-B3 derive confidence from the query latent state, and B3 from scene
    content, but the indicator has no direct query-to-phrase token evidence.
    The decoder uses text earlier, but the confidence score does not expose
    token-level grounding compatibility.

## 4. Candidate Assessment
| Candidate | Hypothesis | Control Comparison | Admission |
|-----------|------------|--------------------|-----------|
| AdaRES-B4 | Predicted-mask overlap among queries supplies relative competition evidence that improves top-1 selection. | Same B1 supervision and data; residual query path preserves candidate independence when masks do not overlap. | Pass |
| AdaRES-B5 | Direct query-to-phrase token compatibility supplies semantic grounding evidence missing from the indicator. | Same B1 supervision and data; padded-token invariance and projection gradients verified. | Pass |

## 5. Rejected Remainder
  - A B3 rerun or mask-pooling ablation is a failed-route repeat.
  - Scalar threshold, temperature, loss-weight, feature-subset, epoch, ranking-
    margin, and occupancy-weight scans are banned.
  - No third slot exists: B4 and B5 consume the two-candidate budget and are
    structurally independent of each other.

## 6. Conclusion
  - Admit AdaRES-B4 on server-2 GPU 0 and AdaRES-B5 on server-2 GPU 1.
  - Both preserve the released SPFormer sparse-UNet backbone, project decoder,
    20-epoch schedule, data split, optimizer, and B1 indicator supervision.
  - Theoretical novelty is not claimed; these are locally motivated causal
    mechanism tests with explicit controls.

  Status: ADMITTED
  Designed by: Research Pipeline (autonomous)
