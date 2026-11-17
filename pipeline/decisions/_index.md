# AdaRES_project 决策索引

| 时间 | 类型 | 问题 | 决策 |
|------|------|------|------|
| 2026-09-18 01:15:00 | [route_change](20260918-011500_route_change.md) | B0 finished at mIoU 0.333; is the failure masks or ranking? | Close B0; rebuild with a faithful standard-indicator baseline |
| 2026-09-18 01:20:00 | [route_change](20260918-012000_route_change.md) | How to avoid another idle period after B1? | Deploy a gated watchdog that smoke-tests and launches B2 after B1 |
| 2026-09-20 09:14:44 | [other](20260920-091444_other.md) | Did the failed B2 smoke invalidate the model? | Fix the model-agnostic smoke harness, revalidate, and launch B2 |
| 2026-09-20 09:20:00 | [manual](20260920-092000_manual.md) | Should new experiments require a user message? | Activate native background goal and successor queue replenishment |
| 2026-09-20 09:37:08 | [next_batch](20260920-093708_next_batch.md) | What successor follows B2 if the target remains unmet? | Admit B3 mask-evidence indicator with conditional automatic launch |
| 2026-09-20 22:40:25 | [manual](20260920-224025_manual.md) | How should server-2 GPUs be allocated after the current B3 run? | Keep B3 unchanged; all later experiments use one independent task per GPU |
| 2026-09-21 11:50:56 | [route_change](20260921-115056_route_change.md) | Does completed B3 replace B2 or justify continuing the same  | Retain B2 as champion; close B3 as completed without demonst |
| 2026-09-21 14:50:43 | [next_batch](20260921-145043_next_batch.md) | Which two structurally independent successors follow B3? | Admit B4 query competition on GPU 0 and B5 token grounding on GPU 1 |
| 2026-09-21 17:54:44 | [route_change](20260921-175444_route_change.md) | Why was B4 running while the scheduler recorded it as waitin | Reconcile B4 as running from verified remote progress; clear |
| 2026-09-21 19:32:33 | [queue_push](20260921-193233_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Token-grounding indicator] configs_song/model_paper5/ada |
| 2026-09-22 08:56:52 | [route_change](20260922-085652_route_change.md) | How should the completed AdaRES-B4 result change the champio | Close B4 as done_complete, promote it to champion at mIoU 0. |
| 2026-09-22 09:30:00 | [manual](20260922-093000_manual.md) | Should the top variants be extended beyond the accidental 20-epoch protocol? | Resume B2 and B4 from epoch 20 to a fixed 30-epoch protocol on separate GPUs |
| 2026-09-22 13:08:09 | [route_change](20260922-130809_route_change.md) | How should AdaRES-B5 be dispositioned after its terminal res | Close AdaRES-B5 at mIoU 0.5030 and do not pursue a duplicate |
| 2026-09-22 19:49:11 | [route_change](20260922-194911_route_change.md) | How should the terminal fixed-protocol B2-E30 and B4-E30 ext | Promote AdaRES-B4-E30 to the best completed result at mIoU 0 |
| 2026-09-22 22:40:02 | [manual](20260922-224002_manual.md) | 陈旧 design tick 长时间运行并阻塞后续调度  | 终止旧 AdaRES tick 进程树，保留 C1/C2 训练，等� |
| 2026-09-23 06:56:48 | [route_change](20260923-065648_route_change.md) | Should the completed C2 hard-negative margin route continue? | Close the tested C2 route; retain B4-E30 champion; no tuning |
| 2026-09-23 08:16:18 | [manual_batch](20260923-081618_manual_batch.md) | 用户要求C1/C2从10 epoch续训至30 epoch  | 分别从各自epoch-10 last.pth恢复优化器与scheduler� |
| 2026-09-23 08:48:54 | [other](20260923-084854_other.md) | How should newly terminal C1 be classified while authorized  | Archive C1 ten-epoch result as below champion; retain B4-E30 |
| 2026-09-24 01:49:16 | [route_change](20260924-014916_route_change.md) | Should the completed C2-E30 hard-negative margin continuatio | Close C2-E30 at 0.5014; retain B4-E30 champion at 0.5094; no |
| 2026-09-24 03:54:21 | [next_batch](20260924-035421_next_batch.md) | One open capacity slot: admit D1 after B4 selection diagnosi | Keep AdaRES-D1 at diagnosis; admit 0 of 1 and leave queue un |
| 2026-09-24 04:53:44 | [other](20260924-045344_other.md) | D1 research validation stage after independent implementatio | Keep D1 unadmitted at validation; queue unchanged  |
| 2026-09-24 05:54:09 | [route_change](20260924-055409_route_change.md) | Close C1-E30 after terminal reconciliation; does D1 pass adm | Close C1-E30 at 0.5008, retain B4-E30 0.5094; D1 remains val |
| 2026-09-24 08:51:48 | [next_batch](20260924-085148_next_batch.md) | Admit and dispatch AdaRES-D1 after six-gate review?  | Six scientific gates pass bounded review, but retain D1 vali |
| 2026-09-24 09:57:03 | [next_batch](20260924-095703_next_batch.md) | Admit D1 after six-gate review and verified source deploymen | Admit one structural D1 candidate and queue on server-2 GPU0 |
| 2026-09-24 10:47:10 | [queue_push](20260924-104710_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Full-query pairwise confidence] configs_song/model_paper |
| 2026-09-24 11:55:11 | [next_batch](20260924-115511_next_batch.md) | Capacity gap 1: admit E1 decoder-mask stability candidate?  | Reject AdaRES-E1 at diagnosis; no queue push or sync; keep s |
| 2026-09-24 14:49:55 | [route_change](20260924-144955_route_change.md) | Does F1 predicted-mask entropy support admission?  | Reject F1 at diagnosis; no implementation or queue admission |
| 2026-09-24 14:50:02 | [route_change](20260924-145002_route_change.md) | Does F2 spatial compactness support admission?  | Reject F2 at diagnosis; no implementation or queue admission |
| 2026-09-24 15:08:57 | [route_change](20260924-150857_route_change.md) | Does F3 seed alignment justify query confidence?  | Reject F3 seed-proximity confidence; 64/240 missed oracles n |
| 2026-09-24 15:09:15 | [next_batch](20260924-150915_next_batch.md) | Does F4 meet six scientific admission gates for server-2 GPU | Admit one F4 decoder reanchor candidate, fixed fresh 30 epoc |
| 2026-09-24 15:10:27 | [queue_push](20260924-151027_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Mask-reanchored decoder neighborhoods] configs_song/mode |
| 2026-09-24 15:13:24 | [manual](20260924-151324_manual.md) | How to ensure D1 and F4 honor fixed 30 epochs under global m | Set task-level min_epochs=30 for both running tasks; retain  |
| 2026-09-25 13:50:53 | [route_change](20260925-135053_route_change.md) | Does F4 predicted-mask centroid reanchoring improve B4-E30?  | Close F4 at 0.5060 after 30/30; 0.0034 below champion; do no |
| 2026-09-25 14:52:23 | [route_change](20260925-145224_route_change.md) | Did D1 full-query pairwise confidence surpass B4-E30 at 30 e | Close D1 at 0.5080 after 30/30, 0.0014 below champion; no re |
| 2026-09-25 17:02:28 | [next_batch](20260925-170228_next_batch.md) | Does metric-aligned G1 indicator supervision merit one singl | Admit AdaRES-G1 on server-2 GPU0 after receipt and safe sync |
| 2026-09-25 17:08:09 | [queue_push](20260925-170809_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-G1 metric-aligned indicator targets] configs_song |
| 2026-09-25 17:29:13 | [next_batch](20260925-172913_next_batch.md) | Does final-layer global query communication merit an indepen | Admit AdaRES-H5 on server-2 GPU1 after receipt and additive  |
| 2026-09-25 17:31:16 | [queue_push](20260925-173116_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-H5 late global query communication] configs_song/ |
| 2026-09-25 17:34:23 | [route_change](20260925-173423_route_change.md) | Which independent routes justify the second capacity slot af | Reject H1 sampler omission, H2 balanced first-layer shortlis |
| 2026-09-25 19:20:49 | [manual](20260925-192049_manual.md) | Enable immediate research when an allowed GPU capacity slot  | Set pipeline_policy.research_on_capacity_gap=true; keep six- |
| 2026-09-26 13:47:22 | [early_stop](20260926-134722_early_stop.md) | [adares_h5_late_global_query_res] 连续 12 次验证未刷新 best，是否早停？ | kill PID=36497 on server-2 gpu=1 |
| 2026-09-26 14:16:47 | [route_change](20260926-141647_route_change.md) | H5 terminal at 0.5061; which mechanism can fill GPU1 under s | Close H5; reject J1 pooled-language contrast and J2 overlap- |
| 2026-09-26 15:04:58 | [route_change](20260926-150458_route_change.md) | Can sampler rank or mask area explain B4 selection errors en | Reject K1 and K2 after native fixed-subset screens; leave se |
| 2026-09-26 16:00:22 | [route_change](20260926-160022_route_change.md) | Can mask-interior coherence or local boundary contrast expla | Reject L1 and L2 after native fixed-subset screens; leave se |
| 2026-09-26 17:03:47 | [route_change](20260926-170348_route_change.md) | Do seed containment or projected-2D appearance support justi | Reject M1 and M2 after native fixed-cohort diagnoses; leave  |
| 2026-09-26 18:33:36 | [route_change](20260926-183336_route_change.md) | G1 best 0.4980 and three distinct confidence diagnoses N1/N2 | Close G1; reject N1 detach-only, N2 inference grouping and N |
| 2026-09-26 19:05:29 | [route_change](20260926-190529_route_change.md) | O1/O2 fixed native counterfactual diagnoses: admit vacant GP | Reject semantic-changing O1 and global-shift O2; leave serve |
| 2026-09-26 20:17:27 | [route_change](20260926-201727_route_change.md) | P1/P2 native training diagnoses: admit either vacant server- | Reject P1 fragmentation and P2 layer swap; leave GPU0/1 unqu |
| 2026-09-26 21:13:58 | [route_change](20260926-211359_route_change.md) | Q1/Q2 native training diagnoses: admit either vacant server- | Reject Q1 token-attention concentration and Q2 attention-mas |
| 2026-09-26 22:04:04 | [route_change](20260926-220404_route_change.md) | R1/R2 fixed native training diagnoses: admit either of two v | Reject oracle-positive filtering R1 and frozen peer-confiden |
| 2026-09-26 23:30:33 | [route_change](20260926-233033_route_change.md) | S1/S2 fixed native train64: admit either of two vacant serve | Reject order-average S1 and local-2D top-two S2; leave serve |
| 2026-09-26 23:59:52 | [route_change](20260926-235952_route_change.md) | T1/T2 fixed native train64: admit either vacant server-2 slo | Reject T1 competitor-removal and T2 local-3D top-two; leave  |
| 2026-09-27 01:00:52 | [route_change](20260927-010053_route_change.md) | U1/U2 fixed native train64: admit either vacant server-2 slo | Reject cross-modal U1 and token-mask U2; leave server-2 GPU0 |
| 2026-09-27 02:00:59 | [route_change](20260927-020059_route_change.md) | V1/V2 fixed native train64: admit either vacant server-2 slo | Reject V1 boundary and V2 query-mask consistency; leave GPU0 |
| 2026-09-27 03:15:06 | [next_batch](20260927-031506_next_batch.md) | W1-W3 fixed train64 and W2 native two-batch controls: admit  | Admit W2 to six-gate receipt and one 30-epoch server-2 singl |
| 2026-09-27 03:22:18 | [queue_push](20260927-032218_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-W2 point-mass competition] configs_song/model_pap |
| 2026-09-29 20:20:46 | [route_change](20260929-202046_route_change.md) | Which distinct inference-available mechanism can close the r | Reject Y7 positive-set ranking and Y8 graph diffusion; do no |
| 2026-09-29 22:42:26 | [route_change](20260929-224226_route_change.md) | Y9/Y10/Y11 three independent inference diagnostics: continue | Reject all three under their preregistered rules; keep Y12 p |
| 2026-09-30 00:47:28 | [route_change](20260930-004728_route_change.md) | Y12/Y13 cross-model committee routes  | Reject Y12 and Y13 under frozen rules; keep Y14 as unrun des |
| 2026-09-30 10:10:46 | [route_change](20260930-101046_route_change.md) | Y14 reliability versus Y15 missing candidates  | Reject frozen Y14; continue Y15 query completion with matche |
| 2026-09-30 12:10:53 | [route_change](20260930-121053_route_change.md) | Y15 native completion admission readiness  | Keep Y15 validation; do not launch full training yet  |
| 2026-09-30 16:43:35 | [route_change](20260930-164335_route_change.md) | Y15完成大场景资源、断点恢复及教师特异性验 | 保持Y15 validation；下轮直接完成源码独立性审� |
| 2026-09-30 19:37:31 | [next_batch](20260930-193731_next_batch.md) | Y15已完成六门证据审查，是否启动一次完整训 | 准入AdaRES-Y15，提交2ccc9051f9ed76b82ffd8a8860d44d6aebb |
| 2026-09-30 19:43:27 | [other](20260930-194327_other.md) | 受控同步被Windows归档换行转换误拦截  | 仅修复project scripts/sync.sh三处archive命令，收据 |
| 2026-09-30 19:50:13 | [queue_push](20260930-195013_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y15 language-conditioned candidate completion] configs_s |
| 2026-09-30 20:26:30 | [route_change](20260930-202630_route_change.md) | Y15 launches; Y16 and Y17 independent screening  | Keep verified Y15 GPU0 training30epochs; reject global Y16 c |
| 2026-09-30 21:33:34 | [next_batch](20260930-213334_next_batch.md) | 是否准入Y17原生30epoch验证  | 准入Y17，GPU1单卡30epoch；保留Y15，不宣称收益  |
| 2026-09-30 23:04:19 | [queue_push](20260930-230419_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y17 target-equivalent paired training] configs_song/mode |
| 2026-09-30 23:08:11 | [other](20260930-230811_other.md) | Y17部署是否恢复并启动  | Y17 GPU1原生递增步数已验证，保留Y15 GPU0；固� |
| 2026-10-01 23:37:51 | [queue_push](20261001-233751_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y18 exact point-measure mask supervision] configs_song/m |
| 2026-10-02 06:17:01 | [route_change](20261002-061701_route_change.md) | Y15 frozen full validation and native query collapse  | Reject Y15 realized efficacy; retain canonical deathwatch an |
| 2026-10-02 08:22:21 | [route_change](20261002-082221_route_change.md) | Y19 Y20 Y21 native diagnoses and capacity gap disposition  | Reject tested realizations; no admissions; evidence-based sh |
| 2026-10-02 11:24:12 | [next_batch](20261002-112412_next_batch.md) | Y23 annotation-valid semantic support versus native padding  | Admit Y23 for fixed30epoch singleGPU0 falsifiable training a |
| 2026-10-02 11:25:58 | [queue_push](20261002-112558_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y23 valid-token semantic support] configs_song/model_pap |
| 2026-10-02 14:46:02 | [manual](20261002-144602_manual.md) | User: pause Y23, free GPU0; resume Y23 on GPU1 after Y18 fin | Y23 paused; GPU0 excluded; guarded GPU1 resume after Y18 epo |
| 2026-10-02 22:06:46 | [other](20261002-220646_other.md) | Y18固定30轮结束后的冻结全量效力与Y23恢复依� | Y18目标未达、场景置信区间跨零；不提升champi |
| 2026-10-02 23:56:15 | [other](20261002-235615_other.md) | Resume SAME Y23 after canonical Y18 completion?  | Resumed original Y23 epoch4 checkpoint on GPU1; PID129584.  |
| 2026-10-03 21:26:46 | [next_batch](20261003-212646_next_batch.md) | Y23 terminal, capacity gap1; are Y25 mechanism/native/resour | Admit only AdaRES-Y25 after bound receipt validation; canoni |
| 2026-10-03 21:28:42 | [queue_push](20261003-212842_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y25 train-only mask-feedback support] configs_song/model |
| 2026-10-04 18:55:21 | [other](20261004-185521_other.md) | Y26 query-language postnorm after Y25 negative terminal  | Advance independent Y26 design to admission review only; no  |
| 2026-10-04 19:29:04 | [question](20261004-192904_question.md) | Y26 mandatory structural smell check omitted in prior review | Do not admit Y26 norm-only realization; preserve all enginee |
| 2026-10-04 19:29:51 | [next_batch](20261004-192951_next_batch.md) | Y27 object-relative peer evidence passes native mechanism an | Admit SAME Y27 for fixed30 falsifiable single-GPU training;  |
| 2026-10-04 19:30:52 | [queue_push](20261004-193052_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y27 mask-centroid relative peer confidence] configs_song |
| 2026-10-05 20:23:04 | [queue_push](20261005-202304_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y28 predicted-mask visual-content readback] configs_song |
| 2026-10-06 19:30:50 | [queue_push](20261006-193050_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y30 ordered query-mask trajectory confidence] configs_so |
| 2026-10-07 19:09:39 | [next_batch](20261007-190939_next_batch.md) | Y30 canonical terminal frees capacity1; does Y31 mask-shape  | Admit only SAME AdaRES-Y31 for fixed30 seed1999 singleGPU1 f |
| 2026-10-07 19:14:04 | [queue_push](20261007-191404_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y31 mask-supported spatial shape confidence] configs_son |
| 2026-10-08 18:23:41 | [question](20261008-182341_question.md) | Y32执行32例实际C1源码对照后，新全query质量通 | 搁置Y32同一实现，不入队；不是数值等价或训 |
| 2026-10-08 18:23:53 | [next_batch](20261008-182353_next_batch.md) | Y31规范终态/容量0of1，Y33六项原生测试及smell� | 仅在提交七个新候选文件、逐字节核验及新CPU |
| 2026-10-08 18:25:48 | [queue_push](20261008-182548_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y33 final mask-conditioned token attention confidence] c |
| 2026-10-09 08:42:10 | [manual](20261009-084210_manual.md) | Authorize server-2 GPU0  | Allow server-2 GPUs [0, 1]; preserve gates and jobs  |
| 2026-10-09 09:11:10 | [question](20261009-091110_question.md) | Is Y34 overlap comparison absent from W2?  | Reject Y34 missing-path realization after actual CPU test  |
| 2026-10-09 09:11:26 | [next_batch](20261009-091126_next_batch.md) | Can Y35 enter one authorized GPU0 slot?  | Admit Y35 fixed30 seed1999 on server2 GPU0; preserve Y33GPU1 |
| 2026-10-09 09:13:03 | [queue_push](20261009-091303_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y35 final mask-evidence language attention confidence] c |
| 2026-10-09 18:32:20 | [other](20261009-183220_other.md) | Y33 frozen terminal scientific disposition  | Reject Y33: preregistered small-object harm versus W2  |
| 2026-10-09 18:32:32 | [next_batch](20261009-183232_next_batch.md) | Y36 six-gate structural admission after native investigation | Admit fixed Y36 for falsifiable training on idle GPU1  |
| 2026-10-09 18:33:37 | [queue_push](20261009-183337_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y36 final mask local-adjacency confidence] configs_song/ |
| 2026-10-10 12:12:53 | [other](20261010-121253_other.md) | Y35 frozen fixed30 terminal scientific disposition  | Close SAME Y35 as inconclusive; retain W2 and preserve check |
| 2026-10-10 12:25:44 | [next_batch](20261010-122544_next_batch.md) | Do actual Y37 gates permit one frozen causal confidence tria | Admit SAME Y37 for fixed30 on GPU0 after exact receipt; pres |
| 2026-10-10 12:27:02 | [queue_push](20261010-122702_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y37 final mask global-pair-distribution confidence] conf |
| 2026-10-10 18:11:58 | [other](20261010-181158_other.md) | Does frozen Y36 training justify target attainment or replac | Same Y36 complete_inconclusive scientifically; retain W2, no |
| 2026-10-10 20:20:11 | [next_batch](20261010-202011_next_batch.md) | Y36已终态且GPU1可用于实验，哪条候选具备真� | 同一Y36科学收尾为complete_inconclusive，不晋升； |
| 2026-10-10 20:21:10 | [queue_push](20261010-202110_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y39 final within-superpoint feature-distribution confide |
| 2026-10-11 16:28:54 | [next_batch](20261011-162854_next_batch.md) | Y37已完成冻结评估，GPU0是否有真正可执行的� | 同一Y37科学收尾为complete_inconclusive，保留W2；Y |
| 2026-10-11 16:29:47 | [queue_push](20261011-162947_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y41 final boundary-probability confidence] configs_song/ |
| 2026-10-11 22:31:24 | [next_batch](20261011-223124_next_batch.md) | Y39终态后Y42完整点协方差是否准入GPU1固定30轮 | Y39证据不充分，保留W2；Y42六项可行性门通过� |
| 2026-10-11 22:32:34 | [queue_push](20261011-223234_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y42 full-channel point covariance confidence] configs_so |
| 2026-10-13 02:32:33 | [next_batch](20261013-023233_next_batch.md) | Y41终态是否晋升；Y43第三阶统计能否进入可证 | Y41保留complete_inconclusive及W2；Y43仅准入Phase1固� |
| 2026-10-13 02:33:51 | [queue_push](20261013-023351_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y43 all-channel point asymmetry confidence] configs_song |
| 2026-10-13 06:29:21 | [next_batch](20261013-062921_next_batch.md) | Y42冻结FIRST为0.504983且对W2场景区间跨零；Y44真 | 同一Y42科学收尾为complete_inconclusive并保留W2；� |
| 2026-10-13 06:30:30 | [queue_push](20261013-063030_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y44 all-coordinate feature-association confidence] confi |
| 2026-10-13 20:08:56 | [manual](20261013-200856_manual.md) | 关机后恢复原训练  | 续训原Y43与Y44，不从头训练  |
| 2026-10-14 12:09:03 | [other](20261014-120903_other.md) | 原生评估与新诊断资源阻塞  | 保留原Y43与同一Y45待GPU释放  |
| 2026-10-14 13:49:42 | [other](20261014-134942_other.md) | 原训练终态和Y45原生验证如何处置  | Y43按冻结小目标伤害拒绝；Y44证据不充分；同 |
| 2026-10-14 14:29:14 | [next_batch](20261014-142914_next_batch.md) | Y45六项可执行性门槛是否足以进入冻结训练验 | 准入同一Y45精确七文件提交；仅在server2 GPU1空 |
| 2026-10-14 14:30:14 | [queue_push](20261014-143014_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y45 point-confusion decomposition confidence] configs_so |
| 2026-10-14 16:09:16 | [next_batch](20261014-160916_next_batch.md) | 容量缺口是否应新增训练，Y47重叠上下文是否 | 拒绝同一Y47缺失机制前提，不更名补位；保留 |
| 2026-10-14 17:19:41 | [next_batch](20261014-171941_next_batch.md) | Y48相同输出掩码的原生标签冲突是否已得到诊 | 保留同一Y48为部分诊断，尚未准入；原生已有 |
| 2026-10-15 11:36:03 | [next_batch](20261015-113603_next_batch.md) | Y45已完成固定30轮且冻结比较为不确定，GPU1自 | 仅准入同一Y48最终硬掩码等价标签闭包，server |
| 2026-10-15 11:37:01 | [queue_push](20261015-113701_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y48 final-mask equivalent targets] configs_song/model_pa |
| 2026-10-15 12:04:43 | [other](20261015-120443_other.md) | Y46独立生产CPU控制已通过，但授权两卡都有真 | 不准入、不排队、不重启。原稳定Y46保留修订 |
| 2026-10-15 13:04:58 | [other](20261015-130458_other.md) | Y49全局图读数的构造源控制与CPU原型是否足以 | 不准入。仅保留同一Y49为有界储备；完成实际 |
| 2026-10-16 13:20:17 | [other](20261016-132017_other.md) | Y48终态和Y46/Y49修订后的实际准入审查 | 拒绝Y48当前监督实现和Y46高成本实现；Y49继续原生信息缺口诊断，不准入、不启动正式训练。 |
| 2026-10-16 14:11:07 | [question](20261016-141107_question.md) | Y49扩样与Y50原生验收之后是否可以正式准入  | Y49当前高成本实现拒绝；Y50验收完成但代码提 |
| 2026-10-16 14:36:59 | [other](20261016-143659_other.md) | Y51公共logit平移是否造成置信度监督与部署选� | 实际native RES已softmax前景概率；65组完整选择� |
| 2026-10-16 14:55:24 | [next_batch](20261016-145524_next_batch.md) | Y50已获得限定提交授权且已测七文件形成真实提交，是否准入同一候选？ | 仅准入AdaRES-Y50高阶共同覆盖置信度，server2单GPU1，released SPFormer/seed19 |
| 2026-10-16 14:57:04 | [queue_push](20261016-145704_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [Y50 joint-incidence confidence fixed30] configs_song/mod |
| 2026-10-16 16:10:31 | [other](20261016-161031_other.md) | Y52辅助质量目标差异和独立CPU原型如何继续？ | 保持Y50原训练；Y52完成CPU准备后真实独占GPU阻塞，不准入、不自动提交。 |
| 2026-10-17 13:28:12 | [question](20261017-132812_question.md) | Y53反证关闭且Y54独立原生门禁完成，是否可以自动提交并派发？ | 不自动提交；Y54仅七个已测文件待人类限定提交许可，保持blocked/human_authorization。Y52既 |
| 2026-10-17 14:08:49 | [other](20261017-140849_other.md) | Y50终态已对账且两项储备仍待提交许可，Y55展 | Y55固定CPU对账反证关闭，不建变体；Y50仅归档 |
| 2026-10-17 14:53:36 | [question](20261017-145336_question.md) | 两项储备未到授权复查时间且储备预算0，为� | 隔离回归已复现共享前台脚本把12条历史closed� |
| 2026-10-17 16:53:55 | [other](20261017-165355_other.md) | Y52四小时提交依赖复查已到期，是否具备解除 | 16:53到期复查：对应限定许可文件仍缺失，原� |
| 2026-10-17 17:54:36 | [other](20261017-175436_other.md) | Y54四小时限定提交依赖复查到期，是否有真实 | 17:53:58到期复查仍缺Y54限定许可，原七文件字� |
| 2026-10-19 12:55:47 | [manual_adares-y52](20261019-125547_manual_adares-y52.md) | AdaRES-Y52已按本会话继续指示提交精确七文件，是否按既有门禁准入？ | 六项有范围限制的既有证据及当前提交已复核，加入server-2规范等待队列；连接超时，未部署或启动。 |
| 2026-10-19 12:55:47 | [manual_adares-y54](20261019-125547_manual_adares-y54.md) | AdaRES-Y54已按本会话继续指示提交精确七文件，是否按既有门禁准入？ | 六项有范围限制的既有证据及当前提交已复核，加入server-2规范等待队列；连接超时，未部署或启动。 |
| 2026-10-19 13:58:40 | [manual](20261019-135840_manual.md) | Only a subset of the authorized GPU indices is currently vis | Remove unavailable soft affinity only; preserve one task per |
| 2026-10-19 13:59:35 | [queue_push](20261019-135935_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y52 fixed30] configs_song/model_paper5/adares_y52 |
| 2026-10-19 14:23:24 | [other](20261019-142324_other.md) | Y54 woke after queue/source changes but server-2 is unreacha | Preserve Y52 registry and the same admitted waiting Y54; awa |
| 2026-10-19 15:11:53 | [other](20261019-151153_other.md) | Storage regression and available CPU reserve budget1. | Preserve Y52/Y54 queue and evidence; record missing original |
| 2026-10-19 15:50:18 | [queue_push](20261019-155018_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y54 fixed30] configs_song/model_paper5/adares_y54 |
| 2026-10-19 16:16:17 | [manual](20261019-161617_manual.md) | User confirms mount restored and requests training. | Preserve interrupted checkpoint-free Y52 attempt and use can |
| 2026-10-19 16:41:55 | [queue_push](20261019-164155_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y52 fixed30] configs_song/model_paper5/adares_y52 |
| 2026-10-19 16:45:10 | [manual](20261019-164510_manual.md) | User confirms mounted storage and requests training. | Both SAME admitted fixed30 candidates verified with increasi |
| 2026-10-19 17:31:54 | [other](20261019-173154_other.md) | Y56 synthetic signal versus actual native-input incidence. | No Y56 implementation/admission from synthetic positives. Re |
| 2026-10-19 19:14:56 | [other](20261019-191456_other.md) | Y57竞争特征均值容量是否构成可训练质量缺陷 | 登记并实测Y57储备；64组输入空间见证不等于� |
| 2026-10-19 20:37:20 | [other](20261019-203720_other.md) | Y57真实GT绑定与描述性关联是否足以实现或准� | 保存32条GT及4组双场景CPU对应结果；错位标签� |
| 2026-10-20 13:03:03 | [manual](20261020-130303_manual.md) | Owning user confirms mount and requests training. | Resume the SAME AdaRES-Y52 from its intact last checkpoint o |
| 2026-10-20 13:03:56 | [manual](20261020-130356_manual.md) | Owning user confirms mount and requests training. | Resume the SAME AdaRES-Y54 from its intact last checkpoint o |
| 2026-10-20 15:49:50 | [manual](20261020-154950_manual.md) | User reports accidental reboot and requests resume. | Continue SAME AdaRES-Y52 from latest verified-attempt last c |
| 2026-10-20 15:50:01 | [manual](20261020-155001_manual.md) | User reports accidental reboot and requests resume. | Continue SAME AdaRES-Y54 from latest verified-attempt last c |
| 2026-10-20 21:24:56 | [next_batch](20261020-212456_next_batch.md) | 继续设计，固定训练已结束，需区分局部机制证据与效益。 | 保留Y56实际复制证据和Y57全部负结果；仅续做同ID原生可达性/全通道二阶矩原型。Y52/Y54完整冻结比较留档，不自 |
| 2026-10-20 22:23:43 | [next_batch](20261020-222343_next_batch.md) | 继续AdaRES原候选与冻结评估，不能以梯度代替科学控制。 | Y52/Y54完整比较闭环，Y54大型目标触发伤害；Y57数值控制实现级暂挂，Y56保留原生损失证据续做。 |
| 2026-10-20 23:11:23 | [manual_adares-y56](20261020-231123_manual_adares-y56.md) | Y56七文件已获本会话明确本地提交授权，是否按既有科学和资源门禁准入？ | 精确测试字节已提交；六项实测门槛复核后只准入Y56，交规范调度器实时验证server-2单卡及原生进度。 |
| 2026-10-20 23:13:51 | [queue_push](20261020-231351_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y56 unique competitors fixed30] configs_song/mode |
| 2026-10-20 23:25:09 | [manual_y57_connection_revision](20261020-232509_manual_y57_connection_revision.md) | Y57晚层二阶矩原数值控制失败，初始证据保留连接是否可修复当前实现？ | 原ID预注册一次连接修订；原4/4失败保留，初始完整256矩跨7层连接的独立实现/CPU/native/资源复测通过，待 |
| 2026-10-20 23:27:49 | [manual_adares-y57](20261020-232749_manual_adares-y57.md) | Y57修订七文件已获单独授权并提交，是否按现有科学及GPU门禁准入？ | 复核六项实测及原失败保留，准入同ID固定30；保持Y56原GPU0/PID，规范调度器只可用实时空闲授权GPU。 |
| 2026-10-20 23:29:14 | [queue_push](20261020-232914_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y57 initial moment anchor fixed30] configs_song/m |
| 2026-10-21 00:19:37 | [next_batch](20261021-001937_next_batch.md) | Y56/Y57 两卡真实训练满载，本轮准入预算0、储备准备预算2；如何推进研究？ | 完成Y58归一化同伴依赖、Y59冻结语言模式两条独立CPU诊断及原生预案；按实占用等待独占空卡，不增加队列或训练。 |
| 2026-10-21 02:00:37 | [next_batch](20261021-020037_next_batch.md) | 实验优先新授权已绑定；已有Y56/Y57真实训练，两项就绪waiting为0。 | 在原Y58/Y59下完成独立包、原生YAML、完整CPU构造/预训练/真实TRAIN拼批/模式实测，精确本地保存14文件 |
| 2026-10-21 02:50:57 | [other](20261021-025057_other.md) | 两项最近可入队的Y58/Y59已实现，如何确保空卡后不会用错版本或被旧任务前提卡住？ | 实测14文件与提交/当前goal/冻结程序一致并拒绝过期goal及篡改源；修复项目检查脚本目录及固定双进程前提，测试归档 |
| 2026-10-21 12:44:52 | [manual](20261021-124452_manual.md) | Owning user requests continued training after GPU reboot. | Resume SAME AdaRES-Y56 from epoch 15 last checkpoint on GPU0 |
| 2026-10-21 12:47:33 | [manual](20261021-124733_manual.md) | Owning user requests continued training after GPU reboot. | Resume SAME AdaRES-Y57 from epoch 17 last checkpoint on GPU1 |
| 2026-10-21 13:13:22 | [next_batch](20261021-131322_next_batch.md) | 待跑不足且Y58/Y59仍待独占GPU，如何推进本轮� | 新增同协议独立Y60阶段专属置信头；CPU原生构 |
| 2026-10-21 15:02:56 | [manual](20261021-150256_manual.md) | GPU0持续90至91C热降频，用户已明确允许仅GPU0� | 核对主机/boot/GPU UUID及原Y56归属后，仅将GPU0运 |
| 2026-10-21 15:06:02 | [other](20261021-150602_other.md) | 三项候选均等待真实独占GPU，如何推进Y60而不 | 保持Y60源码/YAML/提交不变，在CPU执行真实完整 |
| 2026-10-21 20:29:08 | [manual](20261021-202908_manual.md) | 用户要求y57断开，先试用单卡GPU0跑。  | 仅SIGTERM暂停GPU1上原Y57 PID16317及经身份核验的� |
| 2026-10-21 21:01:58 | [other](20261021-210158_other.md) | GPU0单卡试运行时，GPU1出现非AdaRES外部训练负� | 保留Y56/GPU0与280W、Y57暂停及GPU1禁补位；不干� |
| 2026-10-22 01:05:28 | [next_batch](20261022-010528_next_batch.md) | Y56固定30轮已完整归档，唯一授权GPU0已释放，如何续做已实现候选？ | Y58/Y59分别通过当前部署HEAD下原生YAML/真实TRAIN两步有限后向、意图路径及资源检查，按本轮预算2准入； |
| 2026-10-22 01:07:11 | [queue_push](20261022-010711_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y58 fixed30] configs_song/model_paper5/adares_y58 |
| 2026-10-22 11:01:04 | [next_batch](20261022-110104_next_batch.md) | Ready waiting backlog is one; prepare one bounded independen | Implement Y61 peer-value input gradient isolation; retain W2 |
| 2026-10-22 12:55:59 | [other](20261022-125559_other.md) | Y61 component CPU controls passed, but actual seven-stage de | Run and preserve a full-decoder CPU integration on the same  |
| 2026-10-22 22:48:00 | [queue_push](20261022-224800_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y59 fixed30] configs_song/model_paper5/adares_y59 |
| 2026-10-22 22:59:10 | [other](20261022-225910_other.md) | Preflight launched admitted Y59 on physically idle GPU0 whil | Verify full Y58 last/best checkpoints and terminal log, cano |
| 2026-10-23 00:59:19 | [other](20261023-005919_other.md) | 资源复查错误地将已完成Y58的冻结后评待办写成训练前准入，如何修复而不改历史证据？ | 保留旧脚本和本轮原始输出；分离Y56/Y58后评、Y60原生smoke和Y61版本准备。后续资源复查用recheck_g |
| 2026-10-23 20:53:00 | [next_batch](20261023-205300_next_batch.md) | Y59完整固定30轮已归档，Y60是否满足真实检查并接续唯一授权GPU0？ | Y60原生YAML/真实TRAIN两步optimizer及有限loss/backward、七头执行与资源检查通过，按本轮 |
| 2026-10-23 20:54:06 | [queue_push](20261023-205406_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y60 fixed30] configs_song/model_paper5/adares_y60 |
| 2026-10-23 20:58:39 | [other](20261023-205839_other.md) | 空卡采集器已移除retry_after后旧生命周期包装器报KeyError，如何防止记录错误阻断实际接续？ | 保留旧v1/v2和失败；在真实空闲GPU0守卫下完成Y60两步smoke、准入/门控派发和递增步数验证。新v3按生命周期 |
| 2026-10-23 23:03:12 | [other](20261023-230312_other.md) | GPU0被Y60占用时如何推进已通过CPU检查的Y61？ | Y61不再把本地版本保存拖到GPU空闲时：精确CPU已测七文件现已保存，原生两步计划已冻结；真实GPU检查仍只可在新鲜核 |
| 2026-10-24 02:52:48 | [other](20261024-025248_other.md) | Y61已有冻结原生检查计划，但缺自身准入派发交接入口，如何准备？ | 本轮先补齐已准备Y61的原生检查到准入派发交接入口，不借用Y60身份签收据，不改旧hash证据。新入口本地预检通过不能触 |
| 2026-10-24 12:57:56 | [other](20261024-125756_other.md) | Y60训练继续时，如何提前准备其完成后的权威归档交接？ | 趁Y60仍在训练，准备它自己的守卫归档入口与目录绑定计划，不重放Y59 finalizer，不提前归档或伪造deathw |
| 2026-10-24 18:57:11 | [next_batch](20261024-185711_next_batch.md) | Y61是否通过真实原生两步检查并可接续GPU0？ | 原生YAML/TRAIN两步及共享头/peer断开输入/有限梯度/资源检查通过，按预算准入Y61并尝试既有门控派发；Y5 |
| 2026-10-24 18:59:07 | [queue_push](20261024-185907_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y61 fixed30] configs_song/model_paper5/adares_y61 |
| 2026-10-24 19:07:24 | [other](20261024-190724_other.md) | Y60归档残留映射与Y61实际启动后如何交接？ | 保留原归档断言失败，实核后只移除Y60失效映射；Y61原生两步及正式训练步数通过，清理其失效资源字段，四项完成后评等GP |
| 2026-10-24 21:00:26 | [next_batch](20261024-210026_next_batch.md) | Y61运行期间如何准备独立后续对照？ | 准备Y62重叠密度显式输入，CPU原生构造/TRAIN collate/组件及七阶段DEC实测通过，仅七文件本地提交并冻 |
| 2026-10-25 21:56:52 | [next_batch](20261025-215652_next_batch.md) | AdaRES-Y62是否可使用新授权server-1？ | 原生YAML/TRAIN两步、有限梯度及新分支检查通过，按预算准入并使用既有门控派发；Y57保持暂停。 |
| 2026-10-25 21:56:55 | [next_batch](20261025-215655_next_batch.md) | AdaRES-Y63是否可使用新授权server-1？ | 原生YAML/TRAIN两步、有限梯度及新分支检查通过，按预算准入并使用既有门控派发；Y57保持暂停。 |
| 2026-10-25 21:58:03 | [queue_push](20261025-215803_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y62 fixed30] configs_song/model_paper5/adares_y62 |
| 2026-10-25 22:02:00 | [manual](20261025-220200_manual.md) | 用户授权server1并继续训练 | 新增server-1 GPU0；Y62真实两步通过后已确认训练，Y63通过准入并排队；保留Y57暂停、server-2  |
| 2026-10-27 00:51:08 | [queue_push](20261027-005108_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y63 fixed30] configs_song/model_paper5/adares_y63 |
| 2026-10-28 06:53:58 | [next_batch](20261028-065358_next_batch.md) | AdaRES-Y64是否可使用新授权server-1？ | 原生YAML/TRAIN两步、有限梯度及新分支检查通过，按预算准入并使用既有门控派发；Y57保持暂停。 |
| 2026-10-28 06:55:50 | [queue_push](20261028-065550_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y64 fixed30] configs_song/model_paper5/adares_y64 |
| 2026-10-28 10:54:15 | [other](20261028-105415_other.md) | 已有CPU候选如何准备原生检查并保持在跑版本 | 限定提交Y66并冻结原生两步计划；Y65未执行检 |
| 2026-10-29 00:51:43 | [other](20261029-005143_other.md) | server-2持续占用时如何准备已有Y65和Y66的合法s | 保存两项未激活server-1备选计划，原模型和YAML |
| 2026-10-29 04:46:14 | [other](20261029-044614_other.md) | Y64在验证阶段被训练步数观察器误报失败  | 保留旧失败证据并新增Train和Val阶段监测，不� |
| 2026-10-29 06:48:38 | [next_batch](20261029-064838_next_batch.md) | Y65是否通过真实原生两步检查并可接续GPU0？ | 原生YAML/TRAIN两步及六次反馈/七次共享头/实际下游消费/有限梯度及次步响应/资源检查通过，按预算准入Y65并尝 |
| 2026-10-29 06:50:42 | [queue_push](20261029-065042_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y65 fixed30] configs_song/model_paper5/adares_y65 |
| 2026-10-29 10:53:48 | [next_batch](20261029-105348_next_batch.md) | AdaRES-Y66是否可使用新授权server-1？ | 原生YAML/TRAIN两步、六次peer-state反馈/七次共享头、逐点独立公式/实际下游消费和有限非零投影梯度及次 |
| 2026-10-29 10:55:27 | [queue_push](20261029-105527_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y66 fixed30] configs_song/model_paper5/adares_y66 |
| 2026-10-29 17:37:05 | [manual](20261029-173705_manual.md) | 用户重新授权server-2 GPU1 | GPU1加入可用资源，优先推进已准备的Y67；Y57单独暂停仍保留，Y65/Y66原运行不变。 |
| 2026-10-29 17:41:31 | [next_batch](20261029-174131_next_batch.md) | Y67是否通过真实原生两步检查并可接续GPU1？ | 原生YAML/TRAIN两步及七阶段预测归一化/实际mask与score消费/有限非零梯度和真实参数更新/资源检查通过， |
| 2026-10-29 17:43:20 | [queue_push](20261029-174320_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y67 fixed30] configs_song/model_paper5/adares_y67 |
| 2026-10-30 06:53:08 | [next_batch](20261030-065308_next_batch.md) | Y68是否通过真实原生两步检查并可接续GPU0？ | 原生YAML/TRAIN两步及有效语言词均值的mask特征调制/每步七次实际mask预测消费/有限非零梯度及真实参数更新 |
| 2026-10-30 06:54:52 | [queue_push](20261030-065452_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y68 fixed30] configs_song/model_paper5/adares_y68 |
| 2026-10-30 13:16:41 | [next_batch](20261030-131641_next_batch.md) | AdaRES-Y69是否可使用新授权server-1？ | 原生YAML/TRAIN两步、六次粗层几何偏置注意力/七次共享头、原生位置/原hard masks/独立轴公式/实际消费 |
| 2026-10-30 13:18:35 | [queue_push](20261030-131835_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y69 fixed30] configs_song/model_paper5/adares_y69 |
| 2026-10-30 18:45:21 | [next_batch](20261030-184521_next_batch.md) | Y70是否通过真实原生两步检查并可接续GPU1？ | 原生YAML/TRAIN两步及六次查询条件化粗细融合/六真实下游消费/原分支对应及独立公式/有限非零新梯度与真实更新/资 |
| 2026-10-30 18:47:08 | [queue_push](20261030-184708_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y70 fixed30] configs_song/model_paper5/adares_y70 |
| 2026-10-31 07:10:53 | [next_batch](20261031-071053_next_batch.md) | Y71是否通过真实原生两步检查并可接续GPU0？ | 原生YAML/TRAIN两步及一次原3D与2D投影乘积融合/原sum_norm真实消费/原投影数据对应及独立公式/有限非 |
| 2026-10-31 07:12:36 | [queue_push](20261031-071236_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y71 fixed30] configs_song/model_paper5/adares_y71 |
| 2026-10-31 17:07:05 | [next_batch](20261031-170705_next_batch.md) | AdaRES-Y72是否通过真实原生检查并可独占训练？ | 原生YAML/精确TRAIN两步、六新增路径/七共享头/原数据对应与实际预测消费/有限新梯度及参数更新通过，保持原完整后 |
| 2026-10-31 17:09:16 | [queue_push](20261031-170916_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y72 fixed30] configs_song/model_paper5/adares_y72 |
| 2026-10-31 17:10:49 | [next_batch](20261031-171049_next_batch.md) | AdaRES-Y73是否通过真实原生检查并可独占训练？ | 原生YAML/精确TRAIN两步、六新增路径/七共享头/原数据对应与实际预测消费/有限新梯度及参数更新通过，保持原完整后 |
| 2026-10-31 17:12:34 | [queue_push](20261031-171234_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y73 fixed30] configs_song/model_paper5/adares_y73 |
| 2026-11-01 06:56:57 | [next_batch](20261101-065657_next_batch.md) | AdaRES-Y74是否通过真实原生检查并可独占训练？ | 原生YAML/精确TRAIN两步、六原生qla门控/六qqa消费/七共享头/原数据对应/有限新梯度及参数更新通过，保持原 |
| 2026-11-01 06:59:02 | [queue_push](20261101-065902_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y74 fixed30] configs_song/model_paper5/adares_y74 |
| 2026-11-01 15:06:49 | [next_batch](20261101-150649_next_batch.md) | AdaRES-Y75是否通过真实原生检查并可独占训练？ | 原生YAML/精确TRAIN两步、单次原生成query/投影language/padding身份和首次predictio |
| 2026-11-01 15:11:09 | [queue_push](20261101-151109_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y75 fixed30] configs_song/model_paper5/adares_y75 |
| 2026-11-01 20:57:32 | [next_batch](20261101-205732_next_batch.md) | AdaRES-Y76是否通过真实原生检查并可独占训练？ | 原生YAML/精确TRAIN两步、七阶段原query/stage身份与真实query_norm消费/七共享头/七行off |
| 2026-11-01 21:01:26 | [queue_push](20261101-210126_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y76 fixed30] configs_song/model_paper5/adares_y76 |
| 2026-11-02 06:57:14 | [next_batch](20261102-065714_next_batch.md) | AdaRES-Y77是否通过真实原生检查并可独占训练？ | 原生YAML/精确TRAIN两步、单次原query/原scene位置-padding-点数质量及采样坐标身份、十二原SC |
| 2026-11-02 06:59:37 | [queue_push](20261102-065937_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y77 fixed30] configs_song/model_paper5/adares_y77 |
| 2026-11-02 13:02:28 | [next_batch](20261102-130228_next_batch.md) | AdaRES-Y78是否通过真实原生检查并可独占训练？ | 原生YAML/精确TRAIN两步、六原query/raw/previous-mask/padding身份、previou |
| 2026-11-02 13:04:51 | [queue_push](20261102-130451_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y78 fixed30] configs_song/model_paper5/adares_y78 |
| 2026-11-02 23:00:04 | [next_batch](20261102-230004_next_batch.md) | AdaRES-Y79是否通过真实原生检查并可独占训练？ | 原生YAML/精确映射TRAIN两步、六原query/language-FFN/padding身份、未detach与六q |
| 2026-11-02 23:02:19 | [queue_push](20261102-230219_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y79 fixed30] configs_song/model_paper5/adares_y79 |
| 2026-11-03 06:58:26 | [next_batch](20261103-065826_next_batch.md) | AdaRES-Y80是否通过真实原生检查并可独占训练？ | 原生YAML/精确映射TRAIN两步、六原query/visual-FFN/padding身份、未detach与六qia |
| 2026-11-03 07:00:40 | [queue_push](20261103-070040_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y80 fixed30] configs_song/model_paper5/adares_y80 |
| 2026-11-03 11:05:36 | [next_batch](20261103-110536_next_batch.md) | AdaRES-Y81是否通过真实原生检查并可独占训练？ | 原生YAML/精确映射TRAIN两步、六原static mask base/current refined query/ |
| 2026-11-03 11:07:53 | [queue_push](20261103-110753_queue_push.md) | server-2 gpu1 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y81 fixed30] configs_song/model_paper5/adares_y81 |
| 2026-11-04 00:59:45 | [next_batch](20261104-005945_next_batch.md) | AdaRES-Y82是否通过真实原生检查并可独占训练？ | 原生YAML/精确映射TRAIN两步、六原coarse/fine身份、未detach的乘积投影和六qla消费者、七共享c |
| 2026-11-04 01:02:18 | [queue_push](20261104-010218_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y82 fixed30] configs_song/model_paper5/adares_y82 |
| 2026-11-04 06:50:10 | [next_batch](20261104-065010_next_batch.md) | AdaRES-Y83是否通过真实原生检查并可独占训练？ | 原生YAML/精确映射TRAIN两步、六原self-language求和身份/private LayerNorm原eps |
| 2026-11-04 06:52:15 | [queue_push](20261104-065215_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y83 fixed30] configs_song/model_paper5/adares_y83 |
| 2026-11-04 09:36:41 | [manual](20261104-093641_manual.md) | 用户暂时停用server-2 GPU1 | server-2 allowed_gpus收为[0]；GPU1 native/训练/派发/冻结后评暂停，等待明确重新授权 |
| 2026-11-05 03:20:49 | [next_batch](20261105-032049_next_batch.md) | AdaRES-Y84 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-05 03:23:36 | [queue_push](20261105-032336_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y84 fixed30] configs_song/model_paper5/adares_y84 |
| 2026-11-05 05:05:50 | [next_batch](20261105-050550_next_batch.md) | AdaRES-Y85 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-05 05:07:57 | [queue_push](20261105-050757_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y85 fixed30] configs_song/model_paper5/adares_y85 |
| 2026-11-06 03:24:25 | [next_batch](20261106-032425_next_batch.md) | AdaRES-Y87 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-06 03:26:31 | [queue_push](20261106-032631_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y87 fixed30] configs_song/model_paper5/adares_y87 |
| 2026-11-06 05:33:24 | [next_batch](20261106-053324_next_batch.md) | AdaRES-Y88 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-06 05:36:33 | [queue_push](20261106-053633_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y88 fixed30] configs_song/model_paper5/adares_y88 |
| 2026-11-06 11:07:35 | [next_batch](20261106-110735_next_batch.md) | 两张GPU0训练健康，Y89/Y90原生检查受资源阻塞，准备额度仍有空间。 | 在本轮额度内准备独立Y91共享置信度隐状态门控，仅做CPU实现检查和GPU0原生计划，不挤占Y87/Y88。 |
| 2026-11-06 15:02:32 | [other](20261106-150232_other.md) | Y87/Y88在跑，后续候选原生检查需要原运行完整终态核验及归档。 | 只新增绑定两个原始运行的终态检查/归档助手；本轮不加载checkpoint、不停止训练或改变队列。 |
| 2026-11-07 01:23:26 | [next_batch](20261107-012326_next_batch.md) | AdaRES-Y89 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-07 01:25:33 | [queue_push](20261107-012533_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y89 fixed30] configs_song/model_paper5/adares_y89 |
| 2026-11-07 07:12:53 | [next_batch](20261107-071253_next_batch.md) | AdaRES-Y90 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-07 07:14:58 | [queue_push](20261107-071458_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y90 fixed30] configs_song/model_paper5/adares_y90 |
| 2026-11-07 23:20:56 | [next_batch](20261107-232056_next_batch.md) | AdaRES-Y91 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-07 23:23:03 | [queue_push](20261107-232303_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y91 fixed30] configs_song/model_paper5/adares_y91 |
| 2026-11-08 09:00:32 | [next_batch](20261108-090032_next_batch.md) | AdaRES-Y92 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-08 09:02:32 | [queue_push](20261108-090232_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y92 fixed30] configs_song/model_paper5/adares_y92 |
| 2026-11-08 21:25:01 | [next_batch](20261108-212501_next_batch.md) | AdaRES-Y93 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-08 21:27:03 | [queue_push](20261108-212703_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y93 fixed30] configs_song/model_paper5/adares_y93 |
| 2026-11-08 23:26:40 | [next_batch](20261108-232640_next_batch.md) | 两授权GPU0仍被原Y92/Y93占用；补充独立待检查候选并保护Y93精确终态。 | 保留Y94优先；新增Y95全查询均值旁路独立比较，CPU证据及10+8守卫通过后留版；等匹配自然fixed30完整归档和 |
| 2026-11-09 01:12:07 | [next_batch](20261109-011207_next_batch.md) | 两授权GPU0仍被原Y92/Y93占用，本轮快照允许补充独立准备。 | 新增Y96原采样种子到初始查询生成器的线性旁路；原CPU实测及10+8守卫通过后独立留版，Y94/Y95保持优先。 |
| 2026-11-09 03:27:11 | [other](20261109-032711_other.md) | Y97首次CPU检查在精确批次置换逐位守卫失败，如何保留数值事实？ | 保存原程序/日志和相同守卫诊断；Y97按numerical_control_failure/implementation |
| 2026-11-09 05:06:46 | [other](20261109-050646_other.md) | Y97同编号事前登记的逐token执行修订仍未通过完整CPU控制，如何处置？ | 修订1结果记failed并按implementation数值问题挂起；保留原件及修订全部证据，不弱化原守卫、不准入、不重 |
| 2026-11-09 07:21:56 | [next_batch](20261109-072156_next_batch.md) | 授权GPU0继续训练、已有候选未能做原生检查时，如何准备独立结构而不重放旧实验？ | 保留Y94/Y95/Y96优先顺序和Y97数值挂起，准备独立Y98全现有2D外观到静态mask残差；只保存首次CPU通过 |
| 2026-11-09 09:08:10 | [other](20261109-090810_other.md) | 已有Y94/Y95/Y96/Y98通过CPU但授权GPU0仍忙，优先补齐哪个可执行环节？ | 优先为较早Y94准备server1准入前17项新边界检查，保持当前server2激活和双原生计划不变；真实GPU/nat |
| 2026-11-09 11:16:56 | [next_batch](20261109-111656_next_batch.md) | AdaRES-Y94 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-09 11:19:08 | [queue_push](20261109-111908_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y94 fixed30] configs_song/model_paper5/adares_y94 |
| 2026-11-09 11:23:37 | [other](20261109-112337_other.md) | Y92完整终态后，如何推进已通过真实原生检查的Y94并保留Y93？ | 仅精确选择Y94已测七项新增文件，保留Y93原PID/协议，完成完整冻结收据、入队、canonical派发和递增原生co |
| 2026-11-09 13:00:43 | [other](20261109-130043_other.md) | Y93/Y94持续训练且GPU0无空槽时，如何推进较早Y95并完善Y94终态安全？ | Y95同字节科学文件保留到实际native通过的Y94父版本，新增10项执行绑定测试；只读确认Y94原日志保存目录，新增 |
| 2026-11-09 14:54:35 | [other](20261109-145435_other.md) | 已有Y95执行绑定就绪但授权GPU0均忙，下一步可安全准备什么？ | 优先为Y95准备server2准入前只读检查，17项新测试首轮通过，保持当前server2激活和双原生计划不变。 |
| 2026-11-09 17:13:44 | [next_batch](20261109-171344_next_batch.md) | GPU0满载且无ready waiting，如何准备一个与已有掩码路线可区分的独立变体？ | 独立W2保留完整x_mask，增加全部原32维稀疏输入的零初始化线性旁路；稳定ID Y99。首次实际CPU构造/原TRA |
| 2026-11-09 21:18:02 | [next_batch](20261109-211802_next_batch.md) | AdaRES-Y95 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-09 21:20:58 | [queue_push](20261109-212058_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y95 fixed30] configs_song/model_paper5/adares_y95 |
| 2026-11-10 16:54:04 | [next_batch](20261110-165404_next_batch.md) | AdaRES-Y96 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-10 16:57:33 | [queue_push](20261110-165733_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y96 fixed30] configs_song/model_paper5/adares_y96 |
| 2026-11-10 21:04:49 | [next_batch](20261110-210449_next_batch.md) | AdaRES-Y98 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-10 21:08:19 | [queue_push](20261110-210819_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y98 fixed30] configs_song/model_paper5/adares_y98 |
| 2026-11-11 18:04:20 | [next_batch](20261111-180420_next_batch.md) | AdaRES-Y99 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-11 18:07:08 | [queue_push](20261111-180708_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y99 fixed30] configs_song/model_paper5/adares_y99 |
| 2026-11-11 21:51:36 | [next_batch](20261111-215136_next_batch.md) | AdaRES-Y100 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-11 21:54:36 | [queue_push](20261111-215436_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y100 fixed30] configs_song/model_paper5/adares_y1 |
| 2026-11-12 20:00:44 | [next_batch](20261112-200044_next_batch.md) | AdaRES-Y101 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-12 20:03:08 | [queue_push](20261112-200308_queue_push.md) | server-1 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y101 fixed30] configs_song/model_paper5/adares_y1 |
| 2026-11-12 23:53:20 | [next_batch](20261112-235320_next_batch.md) | AdaRES-Y102 native TRAIN admission | Original native YAML/TRAIN two finite backward and optimizer |
| 2026-11-12 23:55:39 | [queue_push](20261112-235539_queue_push.md) | server-2 gpu0 空闲（util<10% 且显存<1024MB），推送哪个排队实验？ | 启动 [AdaRES-Y102 fixed30] configs_song/model_paper5/adares_y1 |
