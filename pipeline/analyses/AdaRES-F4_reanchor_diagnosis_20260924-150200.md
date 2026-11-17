# AdaRES-F4 decoder neighborhood diagnosis

2026-09-24 15:02 Asia/Shanghai. Fixed B4-E30 best checkpoint, first 16
ScanRefer validation scenes (441 expressions), read-only server-2 GPU1 probe.
The F4 intervention is decoder neighborhood reanchoring, not an inference-time
confidence reranker.

The B4 decoder anchors its local fine cross-attention and query self-attention
to immutable sampled seed positions. The probe compared four nearest valid
superpoints around each seed and around the previous predicted-mask soft
centroid against **final predicted foreground** for the missed oracle query.
Among 240 material selection errors, 113 center neighborhoods covered more
of that foreground, 16 covered less; average proxy coverage was 0.14642
(seed) versus 0.23379 (center). There is room for an information-flow
intervention, not evidence of improved trained mIoU.

Limitations: the oracle is chosen with ground truth only for diagnosis, never
at inference. The probe uses all valid superpoints, whereas actual fine
cross-attention selects up to 512 via coarse attention; the query self-attention
also uses a separate top-k mask. It measures overlap with **predicted** final
foreground, not ground-truth foreground, so high coverage may reinforce a
wrong prediction. It is observational, from one trained checkpoint and 16
validation scenes. No causal or population claim follows.

The proposed test changes the attention neighborhood anchor after the initial
decoder layer. It retains B4's scoring indicator, query sampling, top-k sizes,
training loss, seed and 30-epoch schedule. Empty predicted foreground uses
the original seed; padded superpoints are excluded. Training from the
released SPFormer backbone, compare with the B4-E30 protocol control.

Evidence: `pipeline/analyses/AdaRES-F4_reanchor_16scenes_20260924.json`
and `AdaRES/tools/diagnose_adares_f4_reanchor.py`.
