# Motivation Review - AdaRES_project
2026-09-24 03:53 +08:00

## 1. Goal
- "AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."
- B1 0.5020; champion B4-E30 0.5094; target gap 0.0056.
- Constraints: server-2 GPUs 0/1, one experiment per GPU; released SPFormer backbone and project decoder; report mIoU, Acc@0.25/0.5, active queries, latency and peak memory. No validation tuning of thresholds, temperatures, weights, feature subsets, or epoch protocol. No server-6, released b005/CFNet substitution, ranking-margin or occupancy-weight scan.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Candidate masks can outperform chosen query | supported on a bounded subset | B4-E30 16-scene top-1 0.4919, oracle 0.8732 | Full validation oracle unknown |
| Local mask-overlap competition improves confidence | supported at checkpoint level | B4-E30 0.5094 vs B1 0.5020, B2-E30 0.5058 | Mechanism causality and efficiency not isolated |
| Listwise ranking and hard-negative adaptation repair top-1 | failed under tested protocols | C1 0.4950 at 10; C2-E30 0.5014 at 30 | C1-E30 awaits terminal reconciliation |

## 3. Experiment Assessment
| Experiment | mIoU | Interpretation |
|---|---:|---|
| B1 | 0.5020 | Faithful learned-indicator baseline |
| B2-E30 | 0.5058 | Metric-target calibration below B4-E30 |
| B3 | 0.5014 | Mask-conditioned scene evidence did not improve |
| B4-E30 | 0.5094 | Strongest completed local competition route |
| B5 | 0.5030 | Direct token grounding did not improve |
| C1 / C2-E30 | 0.4950 / 0.5014 | Tested ranking losses did not improve champion |

## 4. Gap Analysis
- B4 only aggregates competitors with predicted-mask overlap. On a deterministic 16-scene ScanRefer validation subset (441 expressions), 240 selections lost >0.05 IoU to the available oracle query. Only 49/240 wrong-selected/oracle pairs had mask overlap >=0.5; their mean overlap was 0.1778. This supports inspecting nonlocal query comparisons, not a causal conclusion or a full-validation result.
- The selected/oracle pair need not be the only useful competitor. The bounded diagnostic does not establish whether a nonlocal comparator can learn the right ranking; 20/240 material-error oracles were already score-ranked within three.
- C1-E30 has finished validation but is still queue-running in the preflight; do not make a terminal decision until reconciliation.

## 5. Design Rationale
- Stable candidate AdaRES-D1: compare each query against the entire query set using query/mask pair evidence, including disjoint predicted masks. This changes information flow from B4's overlap-weighted local context, without repeating C1's listwise loss or C2's margin loss.
- Positive result would indicate nonlocal competition helps top-1 selection; negative result would limit this mechanism under the fixed comparison protocol. A matched B4-only adaptation control is required for any causal improvement claim if warm-start adaptation is used.
- No new training admitted today: no independent module/config, native backward execution, matched control, or measured peak memory yet.

## 6. Next Batch
- Candidate budget 1; admitted 0. AdaRES-D1 remains diagnosis/validation work with a concrete next implementation/test action in research_work.json.
- Rejected remainder: overlap-local B4 rework (diagnostic does not support it); C1/C2 repeats, scalar/seed/epoch scans and pure ablations (failed or banned routes). No filler task.
