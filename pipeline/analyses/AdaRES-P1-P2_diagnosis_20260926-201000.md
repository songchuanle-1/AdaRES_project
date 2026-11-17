# Native diagnostic cycle: P1 and P2

2026-09-26 20:10 Asia/Shanghai. Prepared monitor/goal preflight was used once. Server-2 GPU0, key-based SSH, B4-E30 best checkpoint, fixed first 16 training scenes (455 expressions). Both probes are inference-only with repeated-forward controls; neither is a native backward or training-resource test.

## P1: Mask fragmentation

Predeclared topology statistic: largest connected foreground component fraction on the undirected eight-nearest-superpoint spatial graph, binary mask at the existing 0.5 threshold. On 107 material selection errors, the missed oracle was more connected in only 18; mean selected/oracle fractions were 0.94798/0.94565. Topology-only selection gave mIoU 0.53896 versus indicator 0.83083 on this training cohort. Repeat mismatch 0; peak reserved memory 3.46 GiB. This disconfirms fragmentation as a useful direction of query confidence. It is structurally different from earlier radius compactness and feature-boundary probes, but there is no supported trainable mechanism to admit.

## P2: Confidence trajectory

Final versus penultimate indicator was compared against the same final masks. Selection changed in 24/455 expressions, but penultimate selection recovered none of 107 material errors and harmed one otherwise correct expression by >0.05 IoU. Whole-cohort mIoU was 0.8308265 final versus 0.8305341 penultimate. The missed oracle improved its rank in only 9/107 errors. Repeat discrete selection and IoU mismatch 0. An initial exact-dictionary repeat count of 139 was due to float-level margin variation; the corrected tolerance-based control is zero. Peak reserved memory 3.46 GiB. A layer swap is a pure ablation, not an independent confidence mechanism; no layer fusion or validation search is justified.

## Gates and limitations

Diagnosis and mechanism-specific controls fail for both routes. Native backward, full training resources, independent model implementation and admission receipt were not attempted after disconfirmation. Constraint checks prohibit tuning-only or pure-ablation replacements; server-6 remains excluded. The sample is a bounded, high-performing training cohort, not a generalization estimate. Neither theoretical novelty nor efficacy is claimed. No enqueue, sync or dispatch. Resume only on independent inference-available evidence with frozen matched controls and six hard gates; global scientific exhaustion is not asserted.
