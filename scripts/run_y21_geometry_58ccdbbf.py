"""Native training-split geometric counterfactual with immutable evidence."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == "58ccdbbf48104660911c2a0c0fc97d3e"
output = ROOT / "pipeline/analyses/y21_geometry_58ccdbbf.json"
log = output.with_suffix(".log")
assert not output.exists() and not log.exists()
manifest = ROOT / "pipeline/designs/Y21_directional_geometry_prereg_58ccdbbf.md"
program = r'''
import os,sys,pathlib,subprocess,json,fcntl,types,re,statistics
root='/bak_home/songcl/Project/AdaRES_project/AdaRES'
sys.path[:0]=[root,root+'/tools',root+'/ipdn/lib',
'/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages',
'/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg']
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUDA_VISIBLE_DEVICES']='0';os.chdir(root)
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
lease=open('/bak_home/songcl/Project/AdaRES_project/.y15_gpu0_diagnostic.lock','a')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
used,util=[int(v.strip()) for v in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().split(',')]
assert used<1024 and util<10
import numpy as np,torch,gorilla
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class
torch.set_num_threads(4);gorilla.set_random_seed(1999)
config='configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
cfg=gorilla.Config.fromfile(config);data=cfg.data.train.copy()
data.training=False;data.aug=False;data.with_elastic=False;data.lang_num_max=32
dataset=build_dataset(data,get_root_logger());model=resolve_model_class(config)(**cfg.model).cuda()
ck=torch.load('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth',map_location='cpu')
load=model.load_state_dict(ck['model'],strict=False)
assert not load.missing_keys and not set(load.unexpected_keys)-{'text_encoder.embeddings.position_ids','text_encoder.embeddings.token_type_ids'}
del ck;model.eval()
original_transform=dataset.transform_test
def transform(self,xyz,*args):
    xyz=xyz.copy()
    if self.geometry_mode=='rotation': xyz[:,:2]*=-1
    if self.geometry_mode=='translation': xyz+=np.array([1.,1.,0.],dtype=xyz.dtype)
    return original_transform(xyz,*args)
dataset.transform_test=types.MethodType(transform,dataset)
words={'left','right','front','behind','back'}
rows=[];seen=set();scenes=[]
for index,chunk in enumerate(dataset.scene_inputs):
    scan=str(chunk[0]['scene_id'])
    if scan in seen: continue
    seen.add(scan); outputs={};reference=None
    for mode in ['native','repeat','rotation','translation']:
        dataset.geometry_mode=mode
        gorilla.set_random_seed(1999)
        batch=dataset.collate_fn([dataset[index]])
        identity=(batch['ann_ids'],batch['object_idss'])
        if reference is None:
            reference=dict(identity=identity,superpoints=batch['superpoints'].clone(),gt=[v.clone() for v in batch['gt_pmasks']],tokens=batch['lang_tokenss'].clone())
        else:
            assert identity==reference['identity']
            assert torch.equal(batch['superpoints'],reference['superpoints']) and torch.equal(batch['lang_tokenss'],reference['tokens'])
            assert all(torch.equal(a,b) for a,b in zip(batch['gt_pmasks'],reference['gt']))
        with torch.no_grad(): result=model(batch,mode='predict')
        outputs[mode]=dict(iou=[float(v) for v in result['piou']],
            selected=[int(v.argmax()) for v in result['indi']],
            scores=[v.tolist() for v in result['indi']])
        del batch,result
    assert len(chunk)==len(outputs['native']['iou'])
    scene_rows=[]
    for j,entry in enumerate(chunk):
        text=entry.get('description',' '.join(entry.get('token',[])))
        directional=bool(words&set(re.findall(r'[a-z]+',text.lower())))
        base=outputs['native']['iou'][j]
        row=dict(scan_id=scan,ann_id=str(entry['ann_id']),object_id=str(entry.get('object_id',entry.get('object_ids'))),
            directional=directional,native_iou=base,repeat_iou=outputs['repeat']['iou'][j],
            rotation_iou=outputs['rotation']['iou'][j],translation_iou=outputs['translation']['iou'][j],
            rotation_degradation=base-outputs['rotation']['iou'][j],
            translation_degradation=base-outputs['translation']['iou'][j],
            repeat_degradation=base-outputs['repeat']['iou'][j],
            rotation_selection_flip=outputs['native']['selected'][j]!=outputs['rotation']['selected'][j],
            repeat_selection_flip=outputs['native']['selected'][j]!=outputs['repeat']['selected'][j])
        scene_rows.append(row);rows.append(row)
    direction=[r for r in scene_rows if r['directional']]
    other=[r for r in scene_rows if not r['directional']]
    if direction and other:
        scenes.append(dict(scan_id=scan,rotation_contrast=statistics.mean(r['rotation_degradation'] for r in direction)-statistics.mean(r['rotation_degradation'] for r in other),
            translation_contrast=statistics.mean(r['translation_degradation'] for r in direction)-statistics.mean(r['translation_degradation'] for r in other)))
    if len(seen)%8==0: print(json.dumps(dict(scans=len(seen))),flush=True)
    if len(seen)==64: break
summary=dict(scans=len(seen),expressions=len(rows),directional_expressions=sum(r['directional'] for r in rows),
paired_scenes=len(scenes),mean_rotation_contrast=statistics.mean(r['rotation_contrast'] for r in scenes) if scenes else None,
mean_translation_contrast=statistics.mean(r['translation_contrast'] for r in scenes) if scenes else None,
rotation_positive_scenes=sum(r['rotation_contrast']>0 for r in scenes),
rotation_negative_scenes=sum(r['rotation_contrast']<0 for r in scenes),
mean_native_iou=statistics.mean(r['native_iou'] for r in rows),
mean_rotation_iou=statistics.mean(r['rotation_iou'] for r in rows),
mean_translation_iou=statistics.mean(r['translation_iou'] for r in rows),
max_repeat_iou_difference=max(abs(r['repeat_degradation']) for r in rows),
repeat_selection_flips=sum(r['repeat_selection_flip'] for r in rows),
rotation_selection_flips=sum(r['rotation_selection_flip'] for r in rows))
result=dict(candidate='AdaRES-Y21',manifest_sha256='__DIGEST__',summary=summary,rows=rows,scene_contrasts=scenes,complete=True,
limitations=['Direction words are a proxy, view/camera dependence unknown.','Fixed180yaw is not complete augmentation distribution.','TRAIN inference counterfactual, not trained intervention efficacy.'])
pathlib.Path('/tmp/adares_y21_geometry_58ccdbbf.json').write_text(json.dumps(result));print(json.dumps(summary),flush=True)
'''.replace("__DIGEST__", hashlib.sha256(manifest.read_bytes()).hexdigest())
compile(program, "remote_y21_geometry", "exec")
response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "songcl@192.168.1.163", "timeout --signal=TERM --kill-after=10s 480 /home/songcl/.conda/envs/adares_project/bin/python -S -"],
    input=program.encode(), capture_output=True, timeout=510)
log.write_bytes(response.stdout + b"\nSTDERR:\n" + response.stderr)
if response.returncode:
    raise RuntimeError("Failed Y21 geometry; retained " + str(log))
subprocess.run(["scp", "-q", "songcl@192.168.1.163:/tmp/adares_y21_geometry_58ccdbbf.json",
                str(output)], check=True, timeout=60)
result = json.loads(output.read_text())
print(json.dumps(result["summary"], indent=2))
