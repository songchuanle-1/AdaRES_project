"""Read-only final-query decomposition of the original Y15 checkpoint."""

import hashlib
import json
from pathlib import Path
import subprocess

from capture_y15_frozen_terminal import ROOT, TOKEN, CODE, SITE, EGG, PYTHON, Y15


def main():
    owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
    if owner["token"] != TOKEN:
        raise RuntimeError("Owned foreground lease missing or changed")
    output = ROOT / "pipeline/analyses/y15_query_decomposition_v2_963720b2.json"
    log = ROOT / "pipeline/analyses/y15_query_decomposition_v2_963720b2.log"
    if output.exists() or log.exists():
        raise RuntimeError("Refusing to overwrite diagnostic evidence")
    manifest = ROOT / "pipeline/designs/Y15_terminal_query_decomposition_963720b2.md"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    program = r'''
import sys,os,pathlib,json,subprocess,fcntl,hashlib,statistics
sys.path[:0]=__PATHS__
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.chdir(__ROOT__)
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
lease=open('/bak_home/songcl/Project/AdaRES_project/.y15_gpu0_diagnostic.lock','a')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode()
used,util=[int(v.strip()) for v in gpu.strip().split(',')]
assert used<1024 and util<10, gpu
import torch,gorilla
from ipdn.dataset import build_dataset,build_dataloader
from ipdn.utils import get_root_logger
from train import resolve_model_class
torch.set_num_threads(4)
gorilla.set_random_seed(1999)
config='configs_song/model_paper5/adares_y15_query_completion_res.yaml'
cfg=gorilla.Config.fromfile(config)
model=resolve_model_class(config)(**cfg.model).cuda()
checkpoint=torch.load(__CHECKPOINT__,map_location='cpu')
load=model.load_state_dict(checkpoint['model'],strict=False)
assert not load.missing_keys
assert not set(load.unexpected_keys)-{'text_encoder.embeddings.position_ids','text_encoder.embeddings.token_type_ids'}
del checkpoint
model.eval()
dataset=build_dataset(cfg.data.val.copy(),get_root_logger())
loader=build_dataloader(dataset,batch_size=1,num_workers=2,training=False,persistent_workers=True)
captured={}
def hook(module,inputs,out):
    captured['out']=out
handle=model.dec.register_forward_hook(hook)
rows=[]
with torch.no_grad():
    for chunk,batch in enumerate(loader):
        native=model(batch,mode='predict')
        out=captured.pop('out')
        masks=out['masks']
        indicators=out['indis'].softmax(-1)[...,1]
        assert out['completion_index']==128 and masks.shape[1]==129
        for i,scan in enumerate(native['scan_id']):
            truth=native['gt_pmask'][i].cuda().gt(.5)
            sp=batch['superpoints'].cuda().long()
            predicted=masks[i,:,sp].sigmoid().ge(.5)
            intersection=(predicted&truth.unsqueeze(0)).sum(-1)
            union=predicted.sum(-1)+truth.sum()-intersection
            ious=intersection.float()/(union.float()+1e-6)
            selected=int(indicators[i].argmax())
            base=int(indicators[i,:128].argmax())
            native_iou=float(native['piou'][i])
            assert abs(float(ious[selected])-native_iou)<1e-6
            rows.append(dict(scan_id=str(scan),object_ids=[int(v) for v in native['object_ids'][i]],
                ann_id=str(native['ann_id'][i]),selected_index=selected,
                native_iou=native_iou,base_selected_iou=float(ious[base]),
                extra_iou=float(ious[128]),base_oracle_iou=float(ious[:128].max()),
                full_oracle_iou=float(ious.max()),extra_indicator=float(indicators[i,128]),
                base_max_indicator=float(indicators[i,:128].max())))
        del out,masks,indicators
        if (chunk+1)%60==0:
            print(json.dumps(dict(chunks=chunk+1,total=len(loader))),flush=True)
handle.remove()
keys=[(r['scan_id'],tuple(r['object_ids']),r['ann_id']) for r in rows]
assert len(keys)==len(set(keys))
summary={name:statistics.mean(r[name] for r in rows) for name in
    ['native_iou','base_selected_iou','extra_iou','base_oracle_iou','full_oracle_iou']}
summary.update(expressions=len(rows),scenes=len(set(r['scan_id'] for r in rows)),
    extra_selected=sum(r['selected_index']==128 for r in rows),
    extra_selected_fraction=statistics.mean(r['selected_index']==128 for r in rows))
result=dict(candidate='AdaRES-Y15',checkpoint=__CHECKPOINT__,
    checkpoint_sha256=hashlib.sha256(pathlib.Path(__CHECKPOINT__).read_bytes()).hexdigest(),
    manifest_sha256=__MANIFEST__,complete=True,summary=summary,rows=rows,
    limitations=['Diagnostic base-only/oracle is not a deployed method.',
                 'One pass; native identity/IoU matched to separately repeated evaluation.',
                 'Validation diagnosis cannot establish training causal mechanism.'])
pathlib.Path('/tmp/adares_y15_query_decomposition_v2_963720b2.json').write_text(json.dumps(result))
print(json.dumps(summary),flush=True)
'''
    program = program.replace("__PATHS__", repr([CODE, CODE + "/tools", CODE + "/ipdn/lib", SITE, EGG]))
    program = program.replace("__ROOT__", repr(CODE)).replace("__CHECKPOINT__", repr(Y15 + "/best.pth"))
    program = program.replace("__MANIFEST__", repr(digest))
    compile(program, "remote_query_decomposition", "exec")
    response = subprocess.run([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "songcl@192.168.1.163",
        "timeout --signal=TERM --kill-after=10s 480 " + PYTHON + " -S -"
    ], input=program.encode(), capture_output=True, timeout=510)
    log.write_bytes(response.stdout + b"\nSTDERR:\n" + response.stderr)
    if response.returncode:
        raise RuntimeError("Failed query decomposition; see " + str(log))
    subprocess.run(["scp", "-q",
        "songcl@192.168.1.163:/tmp/adares_y15_query_decomposition_v2_963720b2.json",
        str(output)], check=True, timeout=60)
    result = json.loads(output.read_text())
    reference = json.loads((ROOT / "pipeline/analyses/y15_y15_v2_963720b2.json").read_text())
    truth = {(r["scan_id"], tuple(r["object_ids"]), r["ann_id"]): r
             for r in reference["passes"][0]["rows"]}
    assert len(result["rows"]) == len(truth)
    for row in result["rows"]:
        previous = truth[(row["scan_id"], tuple(row["object_ids"]), row["ann_id"])]
        if abs(row["native_iou"] - previous["piou"]) > 1e-6:
            raise RuntimeError("Native readout differs; preserve diagnostic uncertainty")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
