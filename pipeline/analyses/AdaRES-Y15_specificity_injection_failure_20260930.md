# Specificity Probe Injection Failure

Lease a1526671b2ae48f7a80c4f8128bf645a, 2026-09-30.
First specificity launch generated the current teacher masks but model.forward
regenerated them and passed teacher_masks to the physical B4 loss, which
does not accept that keyword. TypeError, no specificity result.
Use student native model.loss directly for this diagnostic's frozen
teacher-output injection. Production data batches do not contain teacher_masks.
Do not infer full forward coverage from this injection; prior native
forward/provider/backward tests establish the online path separately.
