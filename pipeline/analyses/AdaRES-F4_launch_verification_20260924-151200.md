# AdaRES-F4 native launch verification

2026-09-24 15:12 Asia/Shanghai. The project monitor launched the admitted
candidate on server-2 GPU1 after confirming the remote source marker at
597e56a852f1039bc44e72ffcdba6f1a363e07bd. Its dedicated PID file
`AdaRES/logs/.pid_adares_f4_mask_reanchor_res` contains PID 46513.

The exact process command was
`python -S tools/train.py configs_song/model_paper5/adares_f4_mask_reanchor_res.yaml --num_gpus 1 --gpu_ids 1`.
Its unique log is
`AdaRES/logs/train_adares_f4_mask_reanchor_res_20260924_151020.log`.
An immediate read showed `Train[1/30] 0/4686`; a subsequent read after
approximately 48 seconds showed increasing 61/4686 through 67/4686
with learning rate 0.0001 and no observed traceback. This is verified
native training progress, not merely a live PID.

At the later read, GPU0 held 23808 MiB for the separate D1 process and GPU1
held 16180 MiB for F4. Both cards belong to the project whitelist; there is
one independent training run on each GPU. Neither final mIoU nor full-run
peak memory is known. The Windows hourly task
`Pipeline-AdaRES_project` remains enabled, next run 15:47 Asia/Shanghai.
