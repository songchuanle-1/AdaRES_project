# Provider Hardening Regression Plan

2026-09-30 16:12 Asia/Shanghai before hardened-provider outcomes.
Bind the existing B4 checkpoint SHA256
56f49e7c580ddfb67896e77432a922fafed98f7ed58ae6d957703bd2cf062616.
Fail closed if missing/mismatched; no substitute teacher or checkpoint scan.
Isolate Python/NumPy/torch CPU/CUDA RNG around frozen teacher construction and
forward so its execution cannot consume student randomness. Keep current
augmented batch masks and detached material rule unchanged. Audit RNG exact
restoration, teacher device equality and expression object_ids metadata.

Add return self only to the independent Y15/GT train overrides. Preserve W2
and older probes unchanged. Skip provider regeneration when the diagnostic
explicitly supplies teacher_masks; no normal dataset carries that key.

Rerun two released-init native augmented batches/tools.train optimizer updates
with new hashes, inference teacher-free check, wrong-checkpoint rejection,
RNG restoration, native resume and clean largest-batch teacher resources.
Old provider results remain versioned evidence, not the tested new code hash.
GT native resume confirms its return-self change only. No objective,
coefficient, material threshold, query count, epochs or validation tuning.
