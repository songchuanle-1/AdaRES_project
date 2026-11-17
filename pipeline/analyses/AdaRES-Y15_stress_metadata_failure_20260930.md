# Stress Metadata Failure

2026-09-30, lease a1526671b2ae48f7a80c4f8128bf645a.
First launch exited1 before any GPU/model test:
`scene_indices(dataset.scene_inputs, len(dataset.scene_inputs))` raised
`ValueError: Not enough distinct scans`.
The requested count incorrectly used expression chunks rather than unique
scan_ids. Corrected the count to the number of unique scan_id values while
retaining first-chunk-per-scan sampling and the frozen size-selection rule.
No resource result or admission is inferred from this failure.
