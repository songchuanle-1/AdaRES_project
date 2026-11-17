# AdaRES Motivation Review

2026-09-30 20:18 Asia/Shanghai. Review after initial Y16/Y17 screens,
before checkpoint-gradient extension; not backdated upstream evidence.

## 1. Goal

Current goal unchanged: metric mIoU, target.515, dataset ScanRefer,
baseline AdaRES-B1. Description: "AdaRES: improve a faithful
standard-indicator 3D referring expression segmentation baseline by learning
query confidence from metric-aligned evidence, targeting ScanRefer mIoU>=.515."
Constraints: server-2 GPU0/1 only, no server-6; exactly oneGPU and at most
one independent experiment/card; no successor while B3 running, then at
most two structurally independent candidates; fixed30epoch B2/B4 extension
from20 with optimizer/LR preserved; released SPFormer backbone, authored
decoder; report mIoU/Acc25/50/active queries/latency/peak; no val tuning
thresholds/temperatures/loss weights/feature subsets/30epoch protocol.
Bans: no server-6; released b005/CFNet not B0; no val ranking-margin or
occupancy-weight scan. Policy research_on_capacity_gap=true,
steady_ticks_before_design1,fill_to_capacity=true,design_buffer0,
max_candidates_per_design_tick2. W2 .5100, gap.0050.

## 2. Story Status

| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Point-mass competition | historical champion | W2 .5100 | matched causal efficacy |
| Missing complementary candidate | admitted/training | Y15 train64/native gates | trained efficacy/teacher necessity |
| Global within-superpoint output loss | negative | Y16 all-train ceiling .999584 | localized errors not studied |
| Same-target description disagreement | partial | Y17 train16 repeat/control | train dynamics and independent package |

## 3. Experiment Assessment

| Experiment | mIoU | Relation To Goal | Contribution |
|---|---|---|---|
| B1 | .5020 | baseline | faithful native reference |
| W2 | .5100 | competition evidence | current champion |
| Y15 | unknown | extra candidate pool | native launched, not efficacy |
| Y16 | not trained | spatial information discarded | negative global ceiling premise |
| Y17 | not trained | cross-description confidence stability | repeat-separated diagnosis, native conflict |

## 4. Gap Analysis

Y17 selected-mask disagreement is real, but query-mixture JS may induce
generic spatial smoothing instead of learning confidence from target-equivalent
evidence. At initialization input-projection same/wrong gradients are near
collinear and confidence gradients tiny/partly opposite native. Probe trained
state and exact-repeat readout; do not amplify a coefficient to force a result.

## 5. Design Rationale

Continue stable Y17 with a checkpoint-state gradient extension, not a new
family. Read native indicator components and all masks from the SAME
augmented batch. An exact-repeat JS must be zero; same/wrong pairing differs
only by object grouping. Document W2 checkpoint initialization as diagnostic
only, never the production released-backbone initialization.

## 6. Next Batch

Y17 checkpoint-gradient preregistration20260930. No new30epoch task unless
physical implementation, paired sampler, objective rationale, matched controls,
native all-new-path backward and resource gates are completed. Y15 continues.
