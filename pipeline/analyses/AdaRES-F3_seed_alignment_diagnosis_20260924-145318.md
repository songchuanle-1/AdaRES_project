# AdaRES-F3 seed-alignment diagnosis

2026-09-24 14:53 Asia/Shanghai. Read-only B4-E30 best-checkpoint probe,
fixed first 16 ScanRefer validation scenes (441 expressions) on server-2 GPU1.

For each query, F3 measured the distance from its sampled query seed to
the centroid of its binary predicted mask, divided by the scene diagonal.
Among 240 material selection errors, only 64 missed oracles were closer
to their seeds than the selected queries. Mean displacement: selected
0.03559, oracle 0.05334. The predeclared "more seed-aligned means more
reliable confidence" direction is contradicted on this subset.

F3 is rejected as a confidence selector. Do not tune a displacement
threshold or rescore B4 with validation-derived weights. The observation
motivates a distinct **decoder information-flow** hypothesis F4: fixed
seed-centered neighborhoods may miss spatially shifted target masks.
That requires a separate neighborhood-coverage diagnosis and matched
training control, not an inference of benefit from F3's result alone.

Evidence: `pipeline/analyses/AdaRES-F3_seed_alignment_16scenes_20260924.json`
and `AdaRES/tools/diagnose_adares_f3_seed_alignment.py`.
