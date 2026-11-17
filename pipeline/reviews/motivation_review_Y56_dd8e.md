# Motivation Review - AdaRES-Y56

## 1. Goal
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation
baseline by learning query confidence from metric-aligned evidence, targeting
ScanRefer mIoU >= 0.515. Goal and server permissions remain unchanged.

## 2. Story Status
Y56 retains its stable multiplicity-nuisance hypothesis. The frozen trained-W2
diagnostics found exact query+softmask copies only at the initial layer of one
of 40 distinct TRAIN scenes. This is a narrow witnessed mechanism, not evidence
that it explains the overall champion gap or occurs in final-layer predictions.

## 3. Experiment Assessment
The previous real-copy pair8 correction reached actual native task loss and
all seven indicator losses. Its loss delta was -0.000051736831665, repeat
difference 5.96046448e-8; no-copy pair0 remained output-identical. These cached
results justify implementing the mechanism, not a claim of trained efficacy.
Y57 retains its failed numerical control and is not reopened by this work.

## 4. Gap Analysis
Previous inverse-count diagnostic left focal-class copies as competitors.
This first production realization explicitly uses leave-one-PROTOTYPE-out
peers: remove every exact copy of the focal prototype, then weight each other
prototype once. This changes the focal-duplicate boundary and requires new
controls and native tests; previous activity results are not grandfathered.

## 5. Design Rationale
Exact equality of raw query and valid raw mask logits defines classes, without
similarity thresholds or rounding. Membership is detached, the weighted graph
remains differentiable. Keep all W2 inputs, parameters, losses and seven sites.
No-copy rows must be exactly W2; replicated-prototype confidence must match
the same head on unique prototypes within dtype-derived arithmetic error.

## 6. Next Batch
Implement an independent six-Python-file package plus unique YAML. Freeze CPU
controls, actual released-init task backward, trained-W2 case diagnostics and
paired resource tests before execution. No code commit, queue or training is
authorized by this document. Literature novelty remains unverified.
