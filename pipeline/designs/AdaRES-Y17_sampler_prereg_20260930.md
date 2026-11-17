# Y17 Prospective Pairing Coverage And JS Derivative Audit

Frozen2026-09-30 20:22 Asia/Shanghai before new sampler counts.
StableY17, no training admission or mutation of running Y15.

CPU inspect all36665 raw train annotations. Group within scene and exact
target object tuple. Deterministically pair different token strings once,
preserve every annotation exactly once; duplicate strings need different
partners or remain singleton. Order by ann_id, and pack pair units before
singletons into the ORIGINAL per-scene chunk4 sizes. No extra expressions,
scene duplication, annotation deletion or cross-scene pairing. Check identity
multiset, every chunk capacity and whole-scene counts. Compare same-target
eligible pair incidence to the unmodified native annotation order/chunks.
Save annotation hash, pair metadata and exact unavailable singleton counts.
Population bias toward repeated annotated objects remains a limitation;
these counts do not prove effectiveness or sampler/runtime integration.

CPU PyTorch autograd check analytic Bernoulli JS derivatives:
dJS/dp=.5(logit(p)-logit(m)),
dJS/dq=.5(logit(q)-logit(m)),m=(p+q)/2.
For p=sigmoid(z), multiply by p(1-p), then point_mass/total_mass.
Use interior double probabilities with deterministic random points,
finite gradients, closed-form equality against autograd, equal p=q gives
exact zero, exchangingp/q leaves loss invariant. No clamp-edge derivative
claim, native JS still clamps1e-6 for numerical safety.

An actionable implementation must integrate this manifest into a PHYSICAL
independent package/sampler, retain native GT losses, define paired objective
without validation strength search, and verify full gradients/resources.
Do not claim prospective sampler code is the native dataset already in use.
