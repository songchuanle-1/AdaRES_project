# Y21 Native Geometric Diagnosis

- Frozen train64distinctscans/1938expressions,916direction-wordexpressions; all64scanscontainbothgroups.
- Native/repeat/yaw180/translation native W2 inference with point/GT/token/annotation identities checked equal. No optimizer, validation, augmentation/word/threshold scan.
- Primary per-scene rotation degradation direction-minus-other mean=-.0096275429;24positive/40negative. Translation contrast+.0041469448. Native meanIoU.80016907/rotation.80840959/translation.77666987.
- Native repeat maxIoUdifference0/selectionflips0. Rotation changes selectedquery1498/1938; sensitivity is real but not directional-specific harm in the preregistered direction.
- Reject Y21's disproportionate-directional-augmentation degradation rationale. Do not use a favorable subgroup, add words or choose another yaw from this result.
- Translation also reduces meanIoU; this is an unresolved absolute-coordinate/voxelization counterfactual, not proof of the hypothesized augmentation defect. Camera-relative semantics and word-proxy specificity remain unknown.
- Full training efficacy of an orientation-preserving dataset not tested, no admission/resources/native_backward/independence PASS.
- Source dataset transform_train indeed calls jitter/flip/rot=True and text unchanged. Remote source SHA2567b446805d799d1ec8645283fafdb35595b3baa375388645c310ba531947a1b1b; source fact alone is insufficient for candidate admission.
- Evidence: y21_geometry_58ccdbbf.json SHA2566a82655650c92089195e67e6cfb6b5df8cda7d2efdd0d4f1a5b654feb694d185, prereg3018afdade553ba1c83b5a91da72afa2ec32831b71083f388cbf1de72969d1ab, runner49c38147a1ef3c11fad48128e2b305af808019fc2e986e47d0d7bb9b2244fc8e.
- At local2026-10-02T08:15:37+08:00 GPU0device1MiB/util0%, diagnosticadvisoryflockavailable and released; GPU1device22720MiB/util75%. No diagnostic process remains from completed local sessions.
