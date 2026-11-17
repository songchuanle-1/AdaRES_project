"""Frozen gauge controls through exact native predict_by_feat and loss_indi."""
import ast,base64,copy,gc,hashlib,json,types,zlib
from pathlib import Path
from types import SimpleNamespace
import gorilla
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union

assert P['physical_gpu'] is None
torch.set_num_threads(4)
root=Path(P['root'])
package=root/'ipdn/model_paper5/adares_w2_pointmass_competition'
source_record=Path('/tmp/adares_y51_source_fbd1_fbd16ee677ad4d3aa2bbc403acdc2a67.json')
assert hashlib.sha256(source_record.read_bytes()).hexdigest()=='856135f2177705f6f08c58edf6e945337314cedf0c0a7ebebc56752f74a808ce'
source=json.loads(source_record.read_text())
for name,row in source['files'].items():assert hashlib.sha256((package/name).read_bytes()).hexdigest()==row['sha256']
trees={name:ast.parse((package/name).read_text()) for name in source['files']}
parent=next(n for n in trees['ipdn.py'].body if isinstance(n,ast.ClassDef) and n.name=='IPDN')
criterion=next(n for n in trees['loss.py'].body if isinstance(n,ast.ClassDef) and n.name=='Criterion')
predict=next(copy.deepcopy(n) for n in parent.body if isinstance(n,ast.FunctionDef) and n.name=='predict_by_feat')
loss_nodes=[copy.deepcopy(n) for n in criterion.body if isinstance(n,ast.FunctionDef) and n.name in ('loss_indi','_get_src_permutation_idx')]
assert len(loss_nodes)==2
iou=next(copy.deepcopy(n) for n in trees['loss.py'].body if isinstance(n,ast.FunctionDef) and n.name=='get_iou')
exec(compile(ast.Module(body=[predict,iou]+loss_nodes,type_ignores=[]),'<exact-native-predict-and-loss>','exec'),globals())
native_loss_indi=loss_indi
# Native source fixes its one-hot buffer to float32. Only adapt that buffer
# dtype for float64 controls; retain and separately execute the literal loss.
adapted=copy.deepcopy(next(n for n in loss_nodes if n.name=='loss_indi'))
adapted.name='loss_indi_float64'
dtype_changes=[]
for node in ast.walk(adapted):
    if isinstance(node,ast.keyword) and node.arg=='dtype' and isinstance(node.value,ast.Attribute) and isinstance(node.value.value,ast.Name) and node.value.value.id=='torch' and node.value.attr=='float':
        node.value=ast.Attribute(value=ast.Name(id='logits',ctx=ast.Load()),attr='dtype',ctx=ast.Load())
        dtype_changes.append('one_hot_buffer_dtype_only')
assert len(dtype_changes)==1
exec(compile(ast.fix_missing_locations(ast.Module(body=[adapted],type_ignores=[])),'<float64-onehot-dtype-only>','exec'),globals())
dummy=SimpleNamespace(task_type='res',eos_coef=.1)
dummy._get_src_permutation_idx=types.MethodType(_get_src_permutation_idx,dummy)
cfg_path=root/'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
assert hashlib.sha256(cfg_path.read_bytes()).hexdigest()=='d4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb'
cfg=gorilla.Config.fromfile(str(cfg_path))
assert cfg.model.task_type=='res'


def run_predict(logits,masks):
    q,n=masks.shape
    target=torch.zeros(n,dtype=torch.bool)
    return predict_by_feat(dummy,['fixture'],[[0]],['0'],
        dict(masks=[masks],indis=[logits],sample_inds=[torch.arange(q)]),
        torch.arange(n),[target],[target],[torch.arange(q)])


def paired(logits,masks,label):
    q,n=masks.shape
    assert q==128 and logits.dtype==torch.double
    shift=torch.where(torch.arange(q).remainder(2)==0,1.,-1.).double().unsqueeze(1)
    original=logits.detach().clone().requires_grad_(True)
    shifted=(logits+shift).detach().clone().requires_grad_(True)
    indices=[(torch.tensor([0,64]),torch.tensor([0,1]))]
    before=loss_indi_float64(dummy,original.unsqueeze(0),None,indices,2.)
    after=loss_indi_float64(dummy,shifted.unsqueeze(0),None,indices,2.)
    before.backward()
    after.backward()
    with torch.no_grad():
        a,b=run_predict(original,masks),run_predict(shifted,masks)
    bound=8*torch.finfo(torch.double).eps*max(1,n)*q*max(1.,abs(float(before)),abs(float(after)))
    score_error=float((a['indi'][0]-b['indi'][0]).abs().max())
    grad_error=float((original.grad-shifted.grad).abs().max())
    loss_error=abs(float(before)-float(after))
    top_a,top_b=[int(v['indi'][0].argmax()) for v in (a,b)]
    top_two=torch.topk(a['indi'][0],2).values
    top_gap=float(top_two[0]-top_two[1])
    mask_equal=torch.equal(a['pred_pmask'][0],b['pred_pmask'][0])
    raw_a=int(original[:,1].argmax())
    raw_b=int(shifted[:,1].argmax())
    return dict(label=label,nodes=n,loss_error=loss_error,gradient_error=grad_error,
        native_score_error=score_error,existing8eps_nodes_Q_envelope=bound,
        native_top_before=top_a,native_top_after=top_b,native_top_gap=top_gap,
        native_top_equal=top_a==top_b,selected_pointmask_exact=mask_equal,
        native_top_numerically_ambiguous=top_gap<=2*bound,
        raw_foreground_top_before=raw_a,raw_foreground_top_after=raw_b,
        raw_foreground_changed=raw_a!=raw_b)


fixtures=[]
for seed in range(1999,2031):
    torch.manual_seed(seed)
    fixtures.append(paired(torch.randn((128,2),dtype=torch.double),torch.randn((128,19),dtype=torch.double),'seed'+str(seed)))
control=torch.zeros((128,2),dtype=torch.double)
control[:,1]=-4
control[0]=torch.tensor([0.,1.],dtype=torch.double)
control[1]=torch.tensor([1.5,1.5],dtype=torch.double)
positive=paired(control,torch.arange(128*19,dtype=torch.double).reshape(128,19)/1000-1.,'exact_half_integer_positive_control')
assert positive['raw_foreground_changed'] and positive['raw_foreground_top_before']==1 and positive['raw_foreground_top_after']==0
assert positive['native_top_before']==positive['native_top_after']==0
native_a=control.float().unsqueeze(0).requires_grad_(True)
native_shift=torch.where(torch.arange(128).remainder(2)==0,1.,-1.).reshape(1,128,1)
native_b=(native_a.detach()+native_shift).requires_grad_(True)
native_indices=[(torch.tensor([0,64]),torch.tensor([0,1]))]
native_ce_a=native_loss_indi(dummy,native_a,None,native_indices,2.)
native_ce_b=native_loss_indi(dummy,native_b,None,native_indices,2.)
native_ce_a.backward()
native_ce_b.backward()
assert torch.equal(native_ce_a,native_ce_b) and torch.equal(native_a.grad,native_b.grad)

# Reuse previously frozen TRAIN inputs and the exact trained W2 indicator state.
capture_path=Path('/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json')
assert hashlib.sha256(capture_path.read_bytes()).hexdigest()=='5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'
capture=json.loads(capture_path.read_text())
indicator_node=next(copy.deepcopy(n) for n in trees['dec.py'].body if isinstance(n,ast.ClassDef) and n.name=='QueryCompetitionIndicator')
nn=torch.nn
exec(compile(ast.Module(body=[indicator_node],type_ignores=[]),'<exact-native-indicator>','exec'),globals())
checkpoint_path=Path('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth')
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()=='e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
saved=torch.load(str(checkpoint_path),map_location='cpu')
prefix='dec.indi_embedding.'
indicator=QueryCompetitionIndicator(256)
indicator.load_state_dict({k[len(prefix):]:v for k,v in saved['model'].items() if k.startswith(prefix)},strict=True)
indicator=indicator.double().eval()
del saved
gc.collect()


def unpack(item):
    raw=zlib.decompress(base64.b64decode(item['zlib_base64']))
    assert hashlib.sha256(raw).hexdigest()==item['sha256']
    return np.frombuffer(raw,dtype=np.dtype(item['dtype'])).reshape(item['shape']).copy()


native=[]
for scene in capture['rows']:
    for expression in scene['expressions']:
        arrays={key:unpack(value) for key,value in expression['packed_native_inputs'].items() if key in ('query','masks','masses')}
        query=torch.from_numpy(arrays['query'].astype(np.float64)).unsqueeze(0)
        masks=torch.from_numpy(arrays['masks'].astype(np.float64)).unsqueeze(0)
        mass=torch.from_numpy(arrays['masses'].astype(np.float64)).unsqueeze(0)
        pad=torch.zeros((1,masks.shape[-1]),dtype=torch.bool)
        with torch.no_grad():logits=indicator(query,masks,pad,mass)[0]
        row=paired(logits,masks[0],scene['scan_id']+':'+str(expression['ann_id']))
        native.append(row)
assert len(native)==32
all_rows=fixtures+[positive]+native
assert all(max(r['loss_error'],r['gradient_error'],r['native_score_error'])<=r['existing8eps_nodes_Q_envelope'] for r in all_rows)
assert all(r['native_top_equal'] and r['selected_pointmask_exact'] for r in all_rows)
result=dict(complete=True,id='AdaRES-Y51',source_record_sha256=hashlib.sha256(source_record.read_bytes()).hexdigest(),
    source_hashes=source['files'],exact_native_predict_AST=True,configured_task='res',
    literal_native_float32_loss_and_gradient_exact=True,
    float64_loss_instrumentation='Only change source one-hot buffer dtype from hardcoded float32 to logits.dtype; same0/1values, unchangedformula/labels/weights. Literal float32 native loss separately bitwise tested on exact half-integer control.',
    fixture_count=32,native_TRAIN_scenes=8,native_TRAIN_expressions=32,positive_control=positive,
    rows=fixtures,native_rows=native,
    maximum_native_score_error=max(r['native_score_error'] for r in all_rows),
    maximum_native_CE_error=max(r['loss_error'] for r in all_rows),
    maximum_native_gradient_error=max(r['gradient_error'] for r in all_rows),
    native_selected_masks_unchanged=len(all_rows),raw_foreground_changed_cases=sum(r['raw_foreground_changed'] for r in all_rows),
    verdict='Hypothesized common-mode confidence/selection mismatch is disconfirmed for actual RES path: it already uses two-class softmax foreground probability, not raw foreground logit.',
    scientific_status='diagnosis_rejected_not_model_or_efficacy_test',no_model_change=True,no_GPU=True,no_training=True,
    limitations=['This is an exact native confidence/inference consistency audit, not a new calibration model or validation improvement.',
                 'TRAIN inputs and trained indicator are reconstructed in float64 for the frozen numerical control, not a claim of full-loader bitwise equivalence.',
                 'Float64 loss controls require explicit one-hot buffer dtype instrumentation; they are not unmodified full-native execution.',
                 'Zero target masks and fixed positive label indices are fixture bookkeeping only; no task efficacy metric is inferred.',
                 '32native expressions do not establish all numerical robustness or generalization. No ranking/readout family reopened.'])
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
