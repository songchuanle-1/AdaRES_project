# Strict Resume Buffer Compatibility Failure

Lease a1526671b2ae48f7a80c4f8128bf645a, 2026-09-30.
Native checkpoint save and uninterrupted next update completed, but strict
gorilla.resume raised unexpected keys:
text_encoder.embeddings.position_ids and token_type_ids.
Original tools/train.py uses strict=False. Updated the probe to that exact
native mode, with a separate explicit check rejecting all missing keys and
any unexpected key except these two known Roberta nonpersistent buffers.
Exact model/optimizer/scheduler state comparison remains required and is
reported separately; the failed strict assertion is not overwritten.
