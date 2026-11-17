# Native Progress And Y17 Terminal Evidence

- Tick: 3f09f39c723e4389b4b51274b1cd7aba.
- Snapshot: 2026-10-01T12:49:00.596574+00:00; mode event; deadline13:49:00+00.
- Initial live check: 2026-10-01T20:50:06.7000139+08:00.
- Final live check: 2026-10-01T21:19:43.1234024+08:00.
- Server-2 hostname verified dl01-MS-7C60; key-based access only.

| Candidate | GPU | Original PID | Initial / Final Native State | Final Memory MiB / Util |
|---|---|---|---|---|
| Y15 | 0 | 18796 | Train[23/30]1475->1478; later3727->3729/4686 | 17282 / 73% |
| Y17 | 1 | 109632 absent | Val[30/30]363/363, last.pth saved; remains absent | 1 / 0% |

Y15 config and original process remain unchanged; preflight best.2110/latest.2021,
no-improve5/30. Unfavorable history retained; no full Y15 efficacy claim.
Y17 native log reports9508expressions, final mIoU.5014972686767578,
Acc50.551/Acc25.599/spmIoU.467 (latter values rounded in native stdout).
Observed process disappearance plus completed validation and checkpoint metadata
supports training completion; OS exit status of the original launch is unknown.

Y17 last checkpoint:
/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y17_target_equivalent_pairing_res/20260930_103158/last.pth
SHA256795c54f7db7c2df7db763ca7b8ec4961cacc86156f78c514dcf311218d02ea10.
Metadata epoch30/best_miou.5041521191596985; model/optimizer/scheduler present.
Best checkpoint epoch29, SHA2560b931dde7caef19389acc326bf29b0ad51845329b75dcf7984c6c7c6f99548a7.

Queue remains running2 with Y17 deathwatch1/3 in the begin snapshot.
No repeated preflight or queue-state bypass. Real training count at final
bounded check is1; canonical queue will reconcile on later checks.
No new training, admission, scheduler, epoch extension or parameter tuning.

Native full best-checkpoint evaluation completed twice each for Y17/W2/B1,
9508identical expression identities/141scans on live-idle exclusively leasedGPU1.
Standalone utility preserves original native prediction and fixedthresholds.
Its two failures (serialization, B1 inherited registry conflict) and repairs
are retained separately. Each command had900second timeout; all processes exited
before final closure. GPU1 idle after diagnostics.

Raw captures and paired10000draw scene-clustered analysis are separate artifacts.
Y17 mIoU.5041574795120383; W2.5101051604388791; B1.5022242223266429.
VsW2 delta-.005947681,95%CI[-.014310699,.002091696].
VsB1 delta+.001933257,95%CI[-.007090714,.010899705].
All predefined size-group intervals crosszero. Neither improved efficacy nor
material harm established. Target.515 not attained. Exact pair-off causalcontrol,
official unique/multiple labels, broader latency/noise/seed uncertainty remain
unresolved; no novelty/Pareto/effectiveness assertion.

W2 repeats changed4expressions in2scans, maxIoUdelta.9763062,
meanmIoUdelta.0000976621. Largest scene0621_00/object1/ann58:.9763062->0.
Y17 repeats matchedIoU exactly; B1 max.0399251/mean.0000078881.
Two passes are empirical observations, not universal noise bounds.

Training completion is distinct from scientific conclusion and tick completion;
the latter requires finish/abort plus independent lock-release verification.
