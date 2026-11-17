# AdaRES-H5 native gate review

2026-09-25 17:28 Asia/Shanghai. B4-E30 best checkpoint, fixed first 16
ScanRefer validation scenes and two native training pairs on server-2 GPU1.

| Gate | Evidence | Assessment |
|---|---|---|
| Diagnosis | 240 material B4 selection errors; 185 selected/oracle mask pairs overlap <0.5, and 215 pairs are mutually outside final-layer four-nearest query neighborhoods. All are accessible at the first 128-neighbor layer. | Late information-access hypothesis is testable, not a demonstrated causal effect. |
| Controls | Unique H5 config is byte-identical to B4 (SHA256 `1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03`). Dedicated model differs in a parallel all-query attention path only at the final decoder layer. | B4-E30 fixed-protocol control was an epoch-20 optimizer continuation, not a separate fresh matched run. |
| Native backward | Distant-query perturbation changes an unrelated query's global message. Native batches [0,1] and [256,257] produce finite losses and nonzero gradients through new global attention, indicator and decoder. | Pass for tested batches. |
| Resources | H5 [0,1]: 19.72s, 4.775 GiB reserved; B4: 19.63s, 4.750 GiB. H5 [256,257]: 24.11s, 6.406 GiB; B4: 24.01s, 6.396 GiB. | Bounded single-RTX3090 pass; not worst-case training memory. |
| Independence | H5 changes late query-to-query information flow before confidence and mask prediction; G1 changes indicator labels, D1 changes pairwise indicator context, F4 changes spatial retrieval. Unique package/config with no variant imports. | Pass. |
| Constraints | Released backbone, B4 seed/batch/loss, fresh 30 epochs, server-2 GPU1 only, no validation hyperparameter scan. | Pass. |

Native [0,1] global attention gradient norm 0.35753, indicator 2.35935,
decoder 0.23717; [256,257] corresponding norms 0.25505, 1.53828,
0.18819. These are graph and resource tests, not quality estimates.

The final-neighbor rank is based on fixed seed coordinates and is
observational; an earlier full-neighbor layer already allows some nonlocal
communication. H5 could add redundant messages or suppress good local
features. Training efficacy, subgroup harm, noise floor and novelty are
unknown. No validation-derived radius, number of layers or loss weight
is introduced.
