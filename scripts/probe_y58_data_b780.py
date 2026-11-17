"""CPU native-dataset focal identity checks, without constructing a CUDA model."""

import copy
import hashlib
import inspect
import json
from pathlib import Path
import gorilla
import numpy as np
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

assert P['physical_gpu'] is None
torch.set_num_threads(2)
root=Path(P['root'])
assert (root/'.codex_source_commit').read_text().strip()==FROZEN['commit']
for row in FROZEN['sources']:
    assert hashlib.sha256((root/row['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==row['lf_sha256']
cfg=gorilla.Config.fromfile(str(root/'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'))
assert cfg.train.seed==1999 and cfg.train.epochs==30 and cfg.dataloader.train.batch_size==2
assert not cfg.model.norm_eval
dataset=build_dataset(cfg.data.train.copy(),get_root_logger())
source=Path(inspect.getfile(type(dataset)))
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
items=[]
for scene in FROZEN['scenes'][:3]:
    gorilla.set_random_seed(1999)
    item=dataset[scene['dataset_index']]
    single=dataset.collate_fn([copy.deepcopy(item)])
    assert str(single['scan_ids'][0])==scene['scan_id']
    assert [str(v) for v in single['ann_ids']]==scene['ann_ids']
    items.append(item)
focal=dataset.collate_fn([copy.deepcopy(items[0])])
n_points=int(focal['coords_float'].shape[0])
n_expression=len(focal['ann_ids'])


def identical(first,second):
    if torch.is_tensor(first):
        return torch.is_tensor(second) and torch.equal(first,second)
    if isinstance(first,np.ndarray):
        return isinstance(second,np.ndarray) and np.array_equal(first,second)
    if isinstance(first,(list,tuple)):
        return isinstance(second,(list,tuple)) and len(first)==len(second) and all(identical(a,b) for a,b in zip(first,second))
    if isinstance(first,dict):
        return isinstance(second,dict) and first.keys()==second.keys() and all(identical(first[k],second[k]) for k in first)
    return first==second


rows=[]
for mate in (1,2,1):
    batch=dataset.collate_fn(copy.deepcopy([items[0],items[mate]]))
    assert torch.equal(batch['coords_float'][:n_points],focal['coords_float'])
    assert torch.equal(batch['feats'][:n_points],focal['feats'])
    assert batch['scan_ids'][0]==focal['scan_ids'][0]
    assert [str(v) for v in batch['ann_ids'][:n_expression]]==[str(v) for v in focal['ann_ids']]
    identities={}
    for key in ('gt_pmasks','gt_spmasks','sp_ref_masks','object_idss','lang_tokenss','lang_masks'):
        if key not in focal:
            identities[key]='unavailable'
            continue
        first,second=focal[key],batch[key]
        if isinstance(first,(list,tuple)):
            identities[key]=identical(first,second[:len(first)])
        elif torch.is_tensor(first) and torch.is_tensor(second):
            identities[key]=identical(first,second[:first.shape[0]])
        else:
            identities[key]='unsupported layout'
    rows.append(dict(focal=FROZEN['scenes'][0]['scan_id'],mate=FROZEN['scenes'][mate]['scan_id'],
        focal_points=n_points,total_points=int(batch['coords_float'].shape[0]),expressions=n_expression,
        focal_coords_and_features_bitwise_equal=True,focal_annotation_ids_equal=True,
        target_and_language_identity=identities))
assert all(all(value is True for value in r['target_and_language_identity'].values()) for r in rows)
assert not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y58',rows=rows,
    dataset_source=dict(path=str(source),sha256=source_sha),
    dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,
    collate_source_sha256=hashlib.sha256(inspect.getsource(type(dataset).collate_fn).encode('utf-8')).hexdigest(),
    CPU_native_dataset=True,CUDA_initialized=False,native_model_executed=False,
    optimizer_steps=0,weights_saved=False,admission=False,
    limitation='Actual fixed augmented TRAIN focal item/GT/language invariant across two batch mates; '
               'sparse backbone and confidence response still require a free exclusive GPU.')),
    encoding='utf-8')
