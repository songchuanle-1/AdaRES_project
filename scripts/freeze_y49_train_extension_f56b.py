"""Freeze a larger TRAIN-only diagnosis cohort without running new GPU capture."""
import hashlib,json
from pathlib import Path
import gorilla
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
assert P['physical_gpu'] is None
config=Path(P['root'])/'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
assert hashlib.sha256(config.read_bytes()).hexdigest()=='d4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb'
old=Path('/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json')
assert hashlib.sha256(old.read_bytes()).hexdigest()=='5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'
old_ids=[r['scan_id'] for r in json.loads(old.read_text())['rows']]
cfg=gorilla.Config.fromfile(str(config))
dataset=build_dataset(cfg.data.train.copy(),get_root_logger())
seen=[]
chosen=[]
for index,chunk in enumerate(dataset.scene_inputs):
    scene=chunk[0]['scene_id']
    if scene in seen:continue
    seen.append(scene)
    if len(seen)>8:
        chosen.append(dict(dataset_index=index,scan_id=scene,annotation_ids=[r.get('ann_id') for r in chunk[:4]],available_expressions_in_chunk=len(chunk),planned_expressions=min(4,len(chunk))))
    if len(chosen)==24:break
assert seen[:8]==old_ids and len(chosen)==24
result=dict(complete=True,id='AdaRES-Y49',scope='Future TRAIN diagnosis cohort freeze only; new capture NOT executed.',original_TRAIN8_scene_ids=old_ids,additional_TRAIN24=chosen,selection='First24 previously unseen scene IDs in native TRAIN order beyond original first8; first chunk/first4 expressions, or all if fewer. No selection by labels, validation or model outcomes.',seed=1999,queries=128,checkpoint='Existing W2 best; same hash e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52',test='Count actual equal-component-mass-partition but different unit-global-resolvent witnesses under existing8float64eps*nodes bound; retain old zero count, compare actual parent source controls, and report zero honestly if none. Freeze capture program before GPU execution.',limitations=['No new native tensors, gradients, training, admission or task diagnosis PASS from this cohort manifest.','No validation/tuning/model revision; current resource costs and all old negatives retained.'])
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
