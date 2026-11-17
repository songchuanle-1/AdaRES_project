# AdaRES-D1 Native Real-Batch Validation

- Server-2 GPU1, dedicated adares_project Python environment, ScanRefer training dataset, deterministic first two training scenes, seed 1999. No optimizer step or training launch.
- D1 package staged in /tmp/adares_d1_probe_songcl, not synced into production model tree. Probe code: AdaRES/tools/probe_adares_d1_real_batch.py; D1 model: AdaRES/ipdn/model_paper5/adares_d1_full_query_comparison.
- One full model loss forward/backward: finite loss 13.1801; nonzero finite gradient norms for pair_score.0.weight 0.001504, indicator head.0.weight 2.7024 and decoder out_norm.weight 0.6818.
- D1 GPU peak allocated 7.998 GiB, reserved 8.443 GiB; forward/backward 19.661 seconds.
- Same batch, same GPU, B4 package and config: finite loss 13.6225; indicator head gradient norm 2.4766, decoder out_norm 0.6841; peak allocated 4.413 GiB, reserved 4.750 GiB; forward/backward 19.654 seconds.
- D1 reserved peak exceeds B4 by 3.693 GiB. Single cold probes cannot establish steady throughput, inference latency, validation performance, or full training feasibility across all scenes. The 24 GiB device has headroom on this tested batch only.
- Initial B4 probe import failed before model construction due to temporary script path; rerun after adding the repository cwd succeeded. No model changes were made.
- Remaining admission blocker: define and execute a prospectively fixed B4-only matched adaptation control if D1 uses the B4-E30 warm start, with identical split, seed, update count, optimizer and LR schedule. Validate checkpoint loading and record the control receipt before D1 admission/queue push.
