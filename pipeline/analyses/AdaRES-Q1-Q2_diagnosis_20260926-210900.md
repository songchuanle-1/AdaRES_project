# Native diagnostic cycle: Q1 and Q2

2026-09-26 21:09 Asia/Shanghai. Prepared monitor/goal preflight used once. Server-2 GPU0, key SSH, B4-E30 best checkpoint, fixed first 64 training scenes (1660 expressions). Read-only native inference; no training or queue mutation. Both probes repeated each forward. Scene 16-63 is a disjoint within-training check, not validation efficacy.

## Q1: Query-to-token attention

The final query-to-language attention map is computed in B4 but not supplied directly to its competition indicator. Predeclared concentration statistic was squared attention mass summed over valid tokens. In 388 material selection errors, the missed oracle was more concentrated in 260, mean 0.06480 versus selected 0.06173. Concentration-only selection was poor: mIoU 0.08395 versus B4 0.82256. On scenes 16-63, the direction persisted in 186/281 errors, but the attention mass on annotated target tokens favored the oracle in only 147/281; a one-position rotated-token null favored it in 115/281. Full 64-scene target-token counts were 206/388 versus rotated null 160/388. This is a weak, nonspecific signal; positive-map annotations are training-only and cannot be used as inference inputs. The initial exact repeat on 16 scenes differed by 3.7e-8, below the corrected numerical tolerance. On 64 scenes two repeated readouts exceeded 1e-5, maximum 8.67e-5, so strict numerical determinism is not claimed. No attention-feature/fusion-weight scan.

## Q2: Query-to-scene attention support

The final coarse query-to-scene attention map was paired with both the final predicted mask and the previous-layer mask that constrained attention. In 388 material errors, oracle support exceeded selected support in only 70 using the final mask and 25 using the previous mask. Mean final support was 0.91037 oracle versus 0.96527 selected; support-only mIoU 0.55651 versus B4 0.82256. This disconfirms the predicted direction, and previous-mask dependence further weakens an independent evidence claim. One repeated support value exceeded the 1e-4 tolerance despite a maximum reported selected/oracle deviation of 2.86e-6; this implies a discrete selection/oracle mismatch and needs no rescue by tuning.

## Gates and disposition

Q1 has a directional observation but fails a convincing metric-aligned, inference-valid matched-control diagnosis; an annotation-conditioned attention target would leak training supervision into the proposed inference claim. Q2 is directionally disconfirmed. Both are independent from the earlier pooled-mask/text J1 and topology P1 mechanisms, but neither passes diagnosis and controls. Native backward, candidate-specific training resources, independent model implementation, admission receipt, sync and dispatch were not attempted. The inference peak reserved memory was 3.48 GiB (Q1) and 3.56 GiB (Q2), not training peak. Reject both specific routes; no claim of global hypothesis exhaustion.
