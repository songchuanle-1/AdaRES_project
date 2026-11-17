# Prospective native PSD control amendment

Before Y42 upload or any native GPU measurement, add exactly one native
matched control: covariance congruence C -> D C D, with fixed alternating
channel signs D. It is realizable by reflecting point features about their
superpoint means, preserves every mean/diagonal/count/geometry, and preserves
positive semidefiniteness. All32 channels remain used; no setting is selected.
Expected readout response is nonzero, repeats and disabled masks remain zero.

Keep the originally frozen off-diagonal-roll and diagonal-only controls,
including all adverse results. Rolling only off-diagonal entries across
superpoints need not preserve PSD, so that control alone does not prove a
realizable causal feature intervention. The additional fixed congruence
addresses that limitation before observing outcomes, not by relaxing a gate.
Synthetic actual extraction already uses a realizable equal-marginal pair.
