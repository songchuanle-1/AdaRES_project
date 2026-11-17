# Y18 Diagnostic Failure And Repair

- First algebra capture completed1000cases and point-expansion equivalence,
  but original-control helper cast targets to float32, yielding unit-mass
  control error5.90853466153618e-8. Original artifact
  y18_algebra_first_3efff5f3.json retained. Repaired helper casts to logits
  dtype; independent rerun y18_algebra_v2_3efff5f3.json checks1e-12 tolerance,
  unit-mass error2.220446049250313e-16. Candidate loss was not changed.
- First profile command exit1: FileNotFoundError for remote
  /bak_home/songcl/Project/AdaRES_project/y15_stress_manifest_20260930.json.
  Traceback occurred during manifest reading before model construction or GPU
  stress. Process ended, diagnostic lease automatically released.
  Copied existing frozen local all-train metadata to unique remote
  y18_frozen_stress_manifest_3efff5f3.json; source SHA256
  375116a5e035b56f7d81c48ffaa186e589a0c2eb9eab1d2ae4980511ac231f87.
  Rerun writes distinct y18_profile_v2_3efff5f3.json; outcome recorded separately.
- No overwritten evidence, validation-derived sample selection, running
  training restart, threshold relaxation, or successful-profile claim
  from the failed invocation.
- First inference diagnostic exit1 after model forwards: KeyError 'indis'
  while reading the native prediction dictionary. Source returns 'indi'.
  Fixed only the additive diagnostic reader; candidate/native prediction
  source unchanged. Independent rerun uses y18_inference_v2_3efff5f3.json.
