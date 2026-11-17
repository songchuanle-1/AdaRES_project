"""Capture the previously frozen TRAIN24 extension, without optimizer updates."""
import ast,base64,gc,hashlib,json,zlib
from pathlib import Path
import gorilla
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class
assert P['physical_gpu']==1 and torch.cuda.device_count()==1
torch.set_num_threads(4)
reference=Path('/tmp/adares_capture_reference_1cb1.py')
assert hashlib.sha256(reference.read_bytes()).hexdigest()=='2ba5bad46dab8e4528006cbe1357ee9994ff9f18f9f2e8cc30c4444e042184c3'
helpers=[n for n in ast.parse(reference.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('adjacency','components','pack')]
assert len(helpers)==3
exec(compile(ast.Module(body=helpers,type_ignores=[]),str(reference),'exec'),globals())
cohort_path=Path('/tmp/adares_y49_train24_cohort_f56b.json')
assert hashlib.sha256(cohort_path.read_bytes()).hexdigest()=='3cd2f322a628613b4156e55a33b60b26b35bb458801de713937989e3e3e15336'
cohort=json.loads(cohort_path.read_text())
chosen=cohort['additional_TRAIN24']
assert len(chosen)==24 and not set(r['scan_id'] for r in chosen)&set(cohort['original_TRAIN8_scene_ids'])
root=Path(P['root'])
config='configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
assert hashlib.sha256((root/config).read_bytes()).hexdigest()=='d4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb'
protected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'ipdn/model_paper5/adares_w2_pointmass_competition').glob('*.py')}
gorilla.set_random_seed(1999)
cfg=gorilla.Config.fromfile(config)
dataset=build_dataset(cfg.data.train.copy(),get_root_logger())
for row in chosen:
    chunk=dataset.scene_inputs[row['dataset_index']]
    assert chunk[0]['scene_id']==row['scan_id']
    assert [v.get('ann_id') for v in chunk[:4]]==row['annotation_ids']
model=resolve_model_class(config)(**cfg.model).cuda()
checkpoint_path=Path('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth')
checkpoint_sha='e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()==checkpoint_sha
checkpoint=torch.load(str(checkpoint_path),map_location='cpu')
loaded=model.load_state_dict(checkpoint['model'],strict=False)
assert not loaded.missing_keys and not set(loaded.unexpected_keys)-{'text_encoder.embeddings.position_ids','text_encoder.embeddings.token_type_ids'}
del checkpoint
model.train()
capture={}
def dec_pre(module,inputs):
    capture['coordinates']=inputs[5].detach().clone()
    capture['offsets']=inputs[2].detach().clone()
def indicator_pre(module,inputs):
    capture['query'],capture['masks'],capture['padding'],capture['masses']=[v.detach().clone() for v in inputs]
handles=[model.dec.register_forward_pre_hook(dec_pre),model.dec.indi_embedding.register_forward_pre_hook(indicator_pre),model.dec.ila_ffn_layers[-1].register_forward_hook(lambda m,i,o:capture.update(tokens=o.detach().clone()))]
rows=[]
for planned in chosen:
    gorilla.set_random_seed(1999)
    batch=dataset.collate_fn([dataset[planned['dataset_index']]])
    capture.clear()
    with torch.no_grad():loss,_=model(batch,mode='loss')
    assert torch.isfinite(loss) and batch['scan_ids'][0]==planned['scan_id']
    expressions=[]
    assert capture['query'].shape[0]==planned['planned_expressions']
    for b in range(capture['query'].shape[0]):
        start,end=[int(capture['offsets'][b+i]) for i in (0,1)]
        points=capture['coordinates'][start:end]
        valid=~capture['padding'][b]
        assert len(points)==int(valid.sum())
        graph=adjacency(points)
        edges=np.stack(np.nonzero(graph),-1).astype(np.int32)
        arrays=dict(query=capture['query'][b].cpu().numpy(),tokens=capture['tokens'][b,valid].cpu().numpy(),masks=capture['masks'][b,:,valid].cpu().numpy(),masses=capture['masses'][b,valid].cpu().numpy(),coordinates=points.cpu().numpy(),graph_edges=edges,TRAIN_gt_spmask=(batch['gt_spmasks'][b].detach().cpu().reshape(-1).numpy()>.5))
        assert arrays['masks'].shape==(128,len(points))
        expressions.append(dict(ann_id=str(batch['ann_ids'][b]),object_ids=batch['object_idss'][b],graph_nodes=len(points),graph_edges=len(edges),packed_native_inputs={key:pack(value) for key,value in arrays.items()}))
    rows.append(dict(dataset_index=planned['dataset_index'],scan_id=planned['scan_id'],expressions=expressions))
    print('Y49_NEW_TRAIN_SCENE',len(rows),planned['scan_id'],len(expressions),flush=True)
    del batch,loss
    capture.clear()
    gc.collect()
    torch.cuda.empty_cache()
for handle in handles:handle.remove()
assert all(hashlib.sha256((root/path).read_bytes()).hexdigest()==sha for path,sha in protected.items())
result=dict(complete=True,id='AdaRES-Y49',native_scenes=24,native_expressions=sum(len(r['expressions']) for r in rows),queries=128,rows=rows,source_hashes=protected,checkpoint_sha256=checkpoint_sha,cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),capture_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),old_TRAIN8_witness_count=0,new_information_diagnosis='UNTESTED: capture only, not a diagnosis PASS',optimizer_updates=0,limitations=['New TRAIN24 only; original TRAIN8/32 negative evidence unchanged.','Same native parent checkpoint/seed1999/first4 expressions/inherited nearest-tie graph; no validation selection.','TRAIN ground truth captured passively for diagnostics only; never an inference input.','No source/config/checkpoint changed, no optimizer or formal training, no efficacy/novelty/resource admission claim.'])
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
