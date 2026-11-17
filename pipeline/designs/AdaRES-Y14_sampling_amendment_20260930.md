# Y14 Sampling Correction Before Validation

2026-09-30 Asia/Shanghai, token `0eff5f5d4e4847debba4208c85035a3b`.
The first capture attempted 64 dataset items, obtaining only 24 distinct scans.
Fit raised `ValueError: Fit requires exactly train64` before any optimization.
No validation capture or gate fit has occurred at this amendment.

Source inspection: `ipdn/dataset/scannetv2.py` splits a scan's expressions into
items of at most lang_num_max=32. Dataset item indices are not independent scans.
The failed raw capture and log are preserved; never count them as train64.

Correct Y14 sampling prospectively: select the first item of each distinct scan
in existing dataset order, stop at exactly 64 train scans or 32 val scans.
Include all expressions in that first item, at most 32 per scan. Record chosen
item indices and scan IDs. Folds and paired tests group by actual scan_id.
Expression identity is (scan_id, object_ids, ann_id), not ann_id alone.
This correction chooses no observations by their model outcome.
Val32 is now a distinct-scan cohort, not the legacy first-32-item cohort;
do not compare its aggregate directly with Y13's published subset numbers.

All gate features, optimizer settings, epoch count, controls and frozen
acceptance rule remain unchanged. Synthetic sampling/fold tests must pass
before corrected capture. Native duplicate/overlapping scans fail checks.
Prior Y12/Y13 rejections remain unchanged; their item-level "scene" sign tests
are not trusted as independent-scan inferential evidence. A later targeted
statistical audit may document this limitation without reopening those routes.
