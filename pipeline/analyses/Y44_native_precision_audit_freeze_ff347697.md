# Y44 Native Reconstruction Precision Audit Freeze

V2 exited before decoded controls at native_reconstruction_max==0. Actual
max value not printed; cause unknown. Preserve originalv2/source/log. No
model, gate, scene, pairing, optimizer or resource change is authorized.

Before v3 numerical outcomes, capture actual IPDN extract_feat scatter_mean
calls: pointfeature mean, coordinate mean, cross aggregation, originalpool.
Compare actual native cross to independently computed CPUfloat64 reference.
For eachSP/component derive a float32 forward-error envelope from captured
actual mean errors, machine epsilon, perSPpointcount and input magnitudes:
centererror <= measuredmeanerror + eps*(abs(input)+abs(actualmean));
producterror <= abs(centercoord)*featureerror + abs(centerfeature)*coorderror
+ coorderror*featureerror + eps*(abs(centercoord)+coorderror)*
(abs(centerfeature)+featureerror).
Reduction envelope gamma_n=n*eps/(1-n*eps) times mean absolute products;
include divisionroundoff and CPUdouble reference rounding. Verify ALL
components inside this derived envelope; no fitted numeric tolerance.
Also report independent repeatedGPU reconstruction noise and wrong-pairgap.
If native exceeds the envelope, stop and investigate mapping/implementation;
do not admit. If bounded, record numerical identity, not bitwise equality.
Cached native decoded repeats and disabled physicalW2 equality still MUST
be exact0. All eight native finite/nonzero gradient gates remain unchanged.
