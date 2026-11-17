# Y41 CPU arithmetic control audit

The original finite-logit CPU run failed a strict equality assertion and its
raw log is preserved. It is not a PASS. No new model realization or setting is
selected from this failure.

The same 32 finite-logit cases will be retained and printed with their actual
W2/Y36/new-readout differences. In addition, each fixed block/alternating
assignment receives an exactly binary sigmoid control (logits +/-infinity)
solely to diagnose reduction roundoff under the unchanged strict W2 delta=0
assertion. There is no magnitude scan, no validation data and no gate-threshold
change. Binary masks are a synthetic arithmetic control, not a production
input or a backward test.

All finite random-mask gradient, padding, permutation, wrong-input and geometric
controls remain unchanged. A synthetic repaired control still cannot certify
native incidence, native graph activity or trained benefit.
