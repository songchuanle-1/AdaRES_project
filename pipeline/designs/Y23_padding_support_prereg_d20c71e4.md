# Y23 Semantic Padding Support Diagnosis

Owned tick d20c71e418354a6b818d9f37ab72fc7b, deadline11:49:03+08.

## Hypothesis And Controls
Use first8distinct train scans with original augmented train inputs and frozen
W2 checkpoint. Cache actual native criterion inputs. Identity repeat versus
padding-only projected-token replacement by a fixed unit vector; all valid
tokens, annotation maps, query projections, language masks and targets equal.
Execute actual native semantic loss, plus explicit masked-support variant.
Measure padding/valid-token and query gradients, loss nuisance difference and
repeat differences. No validation, weight, temperature, words or feature scan.

## Proposed Intervention
Exclude invalid tokens from query-to-token logsumexp support and exclude their
token-to-query terms. Keep all valid tokens and existing annotation weights.
The full decoder and inference remain unchanged. Support masks are exact data
validity, not a learned gate or a new numerical coefficient.

## Interpretation
Native padding nuisance with removed masked nuisance supports implementation
and further native testing, not trained efficacy. Finite nonzero valid-token
and query gradients are required. If no native padding influence, reject.
Empty-language cases must fail closed; all-valid case must reproduce native.
