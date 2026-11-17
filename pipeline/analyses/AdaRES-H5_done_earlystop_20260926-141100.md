# AdaRES-H5 terminal analysis

## 1. Basic Info
- ID: AdaRES-H5; config: `configs_song/model_paper5/adares_h5_late_global_query_res.yaml`.
- Server-2 GPU1; launched 2026-09-25 17:31:16, reconciled 2026-09-26 13:47:22 Asia/Shanghai.
- State: `done_earlystop`, 30/30 validation epochs recorded. Monitor killed the process after 12 validations without a new best (patience 10, protected until epoch 30).

## 2. Best Metrics
| Metric | Value | Epoch |
|---|---:|---|
| mIoU | 0.5061 | unavailable |
| Acc@0.25 | unavailable | - |
| Acc@0.5 | unavailable | - |
| spmIoU | unavailable | - |

The local terminal record and monitor preserve only rounded best mIoU. The monitor's log pull failed. Active queries, full-run latency and peak memory were not captured. Pretraining native probes measured 4.775 and 6.406 GiB reserved on two paired batches; these are not full-run maxima.

## 3. Convergence Summary
- Best 0.5061, followed by 12 nonimproving validations; final five rounded readouts were 0.5048, 0.5018, 0.4973, 0.5039, 0.5019.
- At 30/30, the process was still partway through training epoch 30 when the protected early-stop rule terminated it. Thus 30 validations do not prove a full final training epoch completed.

## 4. Comparison
- Versus B1 0.5020: +0.0041 mIoU.
- Versus B4-E30 0.5094: -0.0033 mIoU.
- Versus target 0.5150: -0.0089 mIoU.

## 5. Conclusion
The late all-query attention route did not improve the champion in this run. Close H5; do not tune the attention placement or repeat a renamed variant on validation. The pre-run access diagnosis was observational and did not predict the trained benefit. B4-E30 remains champion, while G1 is still running.
