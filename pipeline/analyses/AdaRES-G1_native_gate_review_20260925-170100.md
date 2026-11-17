# AdaRES-G1 native gate review

2026-09-25 17:01 Asia/Shanghai. Fixed B4-E30 checkpoint and first 16
ScanRefer validation scenes (441 expressions), then native training pairs on
server-2 GPU1. No validation-derived threshold or weight is selected.

| Gate | Evidence | Assessment |
|---|---|---|
| Diagnosis | B4 seed/object matched labels disagree with mask quality: among 240 material top-1 errors, 37 selected-positive/oracle-negative pairs. The oracle is positive in 147 errors and the selected query in 54. | Pass as a bounded mismatch; not a causal efficacy claim. |
| Controls | G1 and B4 configs are byte-identical (SHA256 `1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03`); independent package differs only in indicator target computation. | Historical B4-E30 is a fixed 30-epoch continuation control, not a fresh seed-matched run. |
| Native backward | Target test confirms a high-IoU seed-negative query gets positive gradient and a zero-IoU seed-positive query gets negative gradient; target mask detached. Native [0,1] and [256,257] losses finite with nonzero indicator/decoder gradients. | Pass for tested native batches. |
| Resources | G1 [0,1] 19.67s, 4.75 GiB reserved; B4 19.63s, 4.75 GiB. G1 [256,257] 24.05s, 6.396 GiB; B4 24.01s, 6.396 GiB. | Pass bounded one-card 24 GiB budget; not worst-case or final throughput. |
| Independence | G1 changes training indicator target from seed/object membership to detached per-query mask IoU; C1/C2 add listwise/hinge objectives while retaining B4's seed indicator labels, D1 changes rival context, F4 changes decoder neighborhoods. Separate model package and unique config basename. | Distinct causal question, no cross-variant imports. |
| Constraints | Released backbone, same B4 architecture/config, fresh 30 epochs, only server-2 GPU0/1, single-GPU task, no validation scanning. | Pass. |

Native readouts: [0,1] G1 loss 13.39095, indicator/out_norm gradient
norms 3.29086/0.68412, allocated 4.41259 GiB; B4 loss 13.62251,
gradients 2.47664/0.68415. [256,257] G1 loss 10.86048, gradients
2.48718/0.91176, allocated 5.87275 GiB; B4 loss 11.08767,
gradients 1.65840/0.91182. Tests validate mechanism activity, not
generalization or a memory maximum.

The diagnostic reconstructs seed membership from validation instance labels
for analysis only. Training G1's target uses training masks exclusively.
The supervision target depends on current mask quality but is stop-gradient,
so improvement requires learning from changing predictions and could be
unstable. B4-E30 is not a fresh matched training seed control. Literature
novelty and a formal derivation have not been established.
