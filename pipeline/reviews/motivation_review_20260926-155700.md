# Motivation Review - AdaRES_project

2026-09-26 15:57 Asia/Shanghai

## 1. Goal
Goal: "AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515." B1 0.5020, B4-E30 champion 0.5094, gap 0.0056. All seven goal constraints and three bans apply.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Competition-aware confidence | Partial improvement | B4-E30 0.5094 | Not at target; continuation is not a fresh matched control |
| Metric-aligned target labels | Provisional running record | G1 was 0.4983 interim best at prior snapshot; no process in targeted live check | Normal monitor deathwatch and terminal result pending |
| Mask-conditioned visual coherence | Disconfirmed as stated | L1 oracle higher in 116/240 material errors | No admission |
| Mask-local boundary contrast | Disconfirmed as stated | L2 oracle higher in 104/240 material errors | No admission |

## 3. Experiment Assessment
B1 0.5020 and B4-E30 0.5094 establish the baseline and champion. D1 0.5080, F4 0.5060 and H5 0.5061 did not improve the champion; B2-E30 0.5058 and C1/C2-E30 0.5008/0.5014 likewise did not. G1 is not judged by interim metrics or an absent process alone. Earlier J1/J2 and K1/K2 fixed screens failed diagnosis. L1/L2 also fail the proposed directional mechanism.

## 4. Gap Analysis
The fixed B4 subset has oracle mIoU 0.87316 versus selected 0.49186, but this only bounds ranking headroom. Full-validation G1 outcome, active queries, latency and peak memory are pending. No measured confidence mechanism currently passes all six admission gates.

## 5. Design Rationale
L1 tested whether predicted-mask interior visual coherence could supply quality evidence beyond B4's competitor context. L2 tested a different local neighborhood relation at mask boundaries. Both were evaluated against same-scene selected/oracle pairs and standalone selectors before considering a training variant. Negative directions do not justify scanning weights, neighbors, temperatures or feature definitions on validation.

## 6. Next Batch
No new admission, despite candidate budget one. See `pipeline/designs/exp_design_20260926-155700.md`. Resume after G1 terminal reconciliation or on new independent, prospective diagnosis with controls and six gates.
