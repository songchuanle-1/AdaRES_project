# W1-W3 native diagnostic disposition

2026-09-27 03:12 Asia/Shanghai. B4-E30 best checkpoint, fixed first 64
ScanRefer training scenes (1660 expressions), server-2 GPU0, repeated forwards.
These are training-subset diagnoses, not ScanRefer validation estimates.

| ID | Preregistered question | Result | Decision |
|---|---|---|---|
| W1 | Does the penultimate layer contain lost high-quality masks? | Prior oracle exceeds final oracle by >0.05 in 69/1660, all 69 degrade even after nearest-mask matching. Only 23/388 material selection errors overlap. Prior-layer oracle mean 0.91123 versus final 0.91035; fixed-final-query prior mask 0.82344 and prior-layer self-selection 0.82306 versus B4 0.82256. Repeats 0 mismatch. | Reject direct earlier-layer swap: true local degeneration but little usable selection benefit. A new, inference-valid preservation mechanism would require an independent diagnosis. |
| W2 | Does point-count weighting correct B4's superpoint-uniform overlap? | Count CV 3.3245. Real-weight oracle advantage improves relative to B4 for 205/388 missed selections, shifted-count control 166/388. W2 wins direct paired margin against shift 208/388, loses 180/388. Fixed-rule mIoU 0.8225565 versus B4 0.8225620 and shift 0.8226953. Only six W2 selection flips, zero beneficial and one harmful. Repeats 0 mismatch. | Mixed mechanism evidence, not a task gain. Continue only with native matched training-path test, resource and independence checks; prospective outcome remains unknown. |
| W3 | Is the existing mask-quality score conditionally complementary to B4 identity score? | Frozen identity times nonnegative quality 0.82025 versus B4 0.82256; rotated-quality 0.81296. Product changes 951/1660 selections. Oracle margin improvement count 311/388 for both actual and rotated quality, confounded by score scale. Repeats 0 mismatch. | Reject frozen product and refrain from retraining a renamed G1/C1 target; independent calibration diagnosis required to reopen. |

Raw paired rows: `AdaRES-W1_layer_recovery_controls_train64_20260927.json`,
`AdaRES-W2_point_weighted_train64_20260927.json`,
`AdaRES-W3_factorized_train64_20260927.json`. Original preregistered definitions
are in the three `pipeline/designs/AdaRES-W*_prereg_20260927-*.md` files.
W1 first run computed all rows but failed on a nonexistent remote output
directory; it was rerun unchanged to an existing project directory. No
training or queue mutation occurred in these diagnosis runs.
