# Y17 Native Paired-Gradient Preregistration

Frozen2026-09-30 20:15 Asia/Shanghai before gradient outcomes.
Continue stable Y17; diagnosis is not training admission.

Native released-SPFormer initialized W2 graph, original losses/weights,
augmented batch2 with four expressions per scene. Freeze first eight scans
from the earlier prediction-blind Y17 manifest, two same-object real
descriptions for each of two objects. Four batches, seed1999+batch_index.
No decoder checkpoint initialization in this native probe.

Capture final native masks/indicator logits from the same loss forward.
Use differentiable p_e(s)=sum_q softmax_q(indicator_1-indicator_0)
sigmoid(mask_q(s)); query mixing is a training readout, not an inference
replacement or temperature scan. Same-target JS pairs0:1,2:3 versus
different-target pairs0:2,1:3, separately inside each scene.
Point-mass weighted Bernoulli JS, mean across pairs/scenes.
No paired loss is added to production native objective.

Compute gradients of native total, native indicator loss, same-target JS
and wrong-target JS for exactly W2 QueryCompetitionIndicator parameters.
Record norms, cosine/difference, finite reachability, JS values and exact
native losses. Same and wrong controls share the actual augmented tensors,
same native graph and same parameter state. Also measure shared decoder
input_proj gradients to detect actual spatial pathway reachability.
This is partial-parameter local gradient evidence, NOT full optimization,
new-parameter backward, effective convergence or a tested independent variant.
Report adverse/native-opposed gradients and target specificity without
selecting coefficients, cohorts or discarding adverse batches.

Then decide physical objective/package and budget. Missing need for native
paired sampler, coefficient justification and independent package remain
open work; native indicator loss must not be replaced silently.
GPU1 exclusive live-idle only;240seconds timeout; preserve Y15 GPU0.
