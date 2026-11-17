"""Follow-up exact native indicator input audit without replacing prior evidence."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "58ccdbbf48104660911c2a0c0fc97d3e"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN
output = ROOT / "pipeline/analyses/y20_cached_input_58ccdbbf.json"
log = output.with_suffix(".log")
assert not output.exists() and not log.exists()
manifest = ROOT / "pipeline/designs/Y20_cached_input_repeat_58ccdbbf.md"
program = r'''
import os,sys,pathlib,subprocess,fcntl,json,statistics
root='/bak_home/songcl/Project/AdaRES_project/AdaRES'
sys.path[:0]=[root,root+'/tools',root+'/ipdn/lib',
'/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages',
'/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg']
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.chdir(root)
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
lease=open('/bak_home/songcl/Project/AdaRES_project/.y15_gpu0_diagnostic.lock','a')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
used,util=[int(v.strip()) for v in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().split(',')]
assert used<1024 and util<10
import torch,gorilla
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class
torch.set_num_threads(4);gorilla.set_random_seed(1999)
config='configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
cfg=gorilla.Config.fromfile(config);data=cfg.data.train.copy();data.lang_num_max=4
dataset=build_dataset(data,get_root_logger());model=resolve_model_class(config)(**cfg.model).cuda()
ck=torch.load('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth',map_location='cpu')
loaded=model.load_state_dict(ck['model'],strict=False)
assert not loaded.missing_keys and not set(loaded.unexpected_keys)-{'text_encoder.embeddings.position_ids','text_encoder.embeddings.token_type_ids'}
del ck;model.train()
cache={}
def hook(module,inputs,output): cache['inputs']=[v.detach().clone() for v in inputs]
h=model.dec.indi_embedding.register_forward_hook(hook)
seen=set();rows=[]
for i,chunk in enumerate(dataset.scene_inputs):
    scan=chunk[0]['scene_id']
    if scan in seen: continue
    seen.add(scan);batch=dataset.collate_fn([dataset[i]])
    with torch.no_grad(): model(batch,mode='loss')
    query,masks,pad,mass=cache.pop('inputs')
    masks.requires_grad_(True);query.requires_grad_(True)
    h.remove()
    native=model.dec.indi_embedding(query,masks,pad,mass)
    repeat=model.dec.indi_embedding(query,masks,pad,mass)
    detached=model.dec.indi_embedding(query,masks.detach(),pad,mass)
    loss=native.log_softmax(-1)[...,1].mean()
    dl=detached.log_softmax(-1)[...,1].mean()
    ng=torch.autograd.grad(loss,masks,retain_graph=True)[0]
    dg=torch.autograd.grad(dl,masks,retain_graph=True,allow_unused=True)[0]
    hg=torch.autograd.grad(dl,model.dec.indi_embedding.head[0].weight)[0]
    rows.append(dict(scan_id=scan,repeat_logit_max=float((native-repeat).abs().max()),
        detached_logit_max=float((native-detached).abs().max()),native_mask_gradient_norm=float(ng.norm()),
        detached_mask_gradient_absent=dg is None,detached_head_gradient_norm=float(hg.norm()),
        all_finite=bool(torch.isfinite(ng).all()) and bool(torch.isfinite(hg).all())))
    del native,repeat,detached,query,masks,ng,dg,hg,loss,dl,batch
    h=model.dec.indi_embedding.register_forward_hook(hook)
    if len(rows)==8: break
h.remove()
summary=dict(scans=len(rows),maximum_repeat_logit_difference=max(r['repeat_logit_max'] for r in rows),
maximum_detached_logit_difference=max(r['detached_logit_max'] for r in rows),
all_detached_mask_gradient_absent=all(r['detached_mask_gradient_absent'] for r in rows),
all_native_mask_gradient_nonzero=all(r['native_mask_gradient_norm']>0 for r in rows),
all_detached_head_gradient_nonzero=all(r['detached_head_gradient_norm']>0 for r in rows),
all_finite=all(r['all_finite'] for r in rows))
pathlib.Path('/tmp/adares_y20_cached_input_58ccdbbf.json').write_text(json.dumps(dict(summary=summary,rows=rows,manifest_sha256='__DIGEST__',complete=True)))
print(json.dumps(summary),flush=True)
'''.replace("__DIGEST__", hashlib.sha256(manifest.read_bytes()).hexdigest())
compile(program, "remote_cached_input", "exec")
response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "songcl@192.168.1.163", "timeout --signal=TERM --kill-after=10s 240 /home/songcl/.conda/envs/adares_project/bin/python -S -"],
    input=program.encode(), capture_output=True, timeout=270)
log.write_bytes(response.stdout + b"\nSTDERR:\n" + response.stderr)
if response.returncode:
    raise RuntimeError("Failed cached-input audit; retained " + str(log))
subprocess.run(["scp", "-q", "songcl@192.168.1.163:/tmp/adares_y20_cached_input_58ccdbbf.json",
                str(output)], check=True, timeout=60)
result = json.loads(output.read_text())
previous = json.loads((ROOT / "pipeline/analyses/y19_y20_diagnosis_58ccdbbf.json").read_text())
rows = previous["rows"]
result["original64_repeat"] = {
    "mean_native_cosine": sum(r["native"]["cosine"] for r in rows) / len(rows),
    "mean_repeat_cosine": sum(r["repeat"]["cosine"] for r in rows) / len(rows),
    "maximum_cosine_difference": max(abs(r["native"]["cosine"] - r["repeat"]["cosine"]) for r in rows),
    "cosine_sign_changes": sum((r["native"]["cosine"] < 0) != (r["repeat"]["cosine"] < 0) for r in rows),
    "maximum_indicator_loss_difference": max(abs(r["native"]["indicator_loss"] - r["repeat"]["indicator_loss"]) for r in rows),
    "source_sha256": hashlib.sha256((ROOT / "pipeline/analyses/y19_y20_diagnosis_58ccdbbf.json").read_bytes()).hexdigest(),
}
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
