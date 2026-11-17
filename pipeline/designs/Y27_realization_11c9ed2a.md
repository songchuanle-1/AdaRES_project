# Y27 Implementation Freeze

Written before physical controls/backward/resource outcomes.
The relative peer graph implements the preregistration exactly, with existing
d_model256. Projection q/k/v/output biases are omitted: key bias cancels row
softmax, value/output biases would add no peer-specific information. Position
MLP retains its two biases. Thus there are EIGHT, not the preregistration's
estimated twelve, added trainable tensors, all required to have real finite
nonzero native gradients. This is a structural implementation clarification,
not a post-result relaxation. No learned scalar, numerical scan or config edit.
The original query normalizer and confidence-head dimensions remain unchanged.
The existing parent qla normalization bug is deliberately NOT fixed here.
Evidence/readout identity, frozen sample manifest and all adverse outcomes are
preserved. No claim of novel attention algorithm or trained efficacy.
