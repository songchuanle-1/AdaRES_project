# AdaRES Project Environment

## Server

- Host: `server-2`
- Environment name: `adares_project`
- Prefix: `/home/songcl/.conda/envs/adares_project`
- Python: `/home/songcl/.conda/envs/adares_project/bin/python`
- Python version: 3.8.13

## Creation

Created on 2026-09-20 by cloning the previously validated shared environment:

```bash
/home/dl01/anaconda3/bin/conda create -y \
  -p /home/songcl/.conda/envs/adares_project \
  --clone /home/songcl/.conda/envs/project_cluster
```

The running AdaRES-B3 process was not interrupted during environment creation.
New launches and resumes use the dedicated environment.

## Verification

- Conda package count: 254, matching the source environment.
- Pip package count: 241, matching the source environment.
- PyTorch: 1.12.1+cu113.
- CUDA runtime reported by PyTorch: 11.3.
- CUDA available: true; two GPUs detected.
- `spconv.pytorch` import passed.
- `pointnet2` import passed.
- `ipdn.model_paper5.adares_b3_mask_evidence.IPDN` import passed.

The PointNet2 extension remains loaded from:

`/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg`

`pip check` reports platform metadata warnings for `ninja`, `pillow`, and
`tensorboard-data-server`. The source `project_cluster` environment reports
the same warnings with the same package versions and locations. They are
inherited metadata warnings rather than a clone-specific dependency change;
runtime imports and CUDA execution checks passed.
