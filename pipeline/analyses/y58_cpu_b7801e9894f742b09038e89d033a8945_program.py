PAYLOAD={'commit': '05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea', 'sources': [{'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/__init__.py', 'lf_sha256': '8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/backbone.py', 'lf_sha256': '1eb4ac3483961e708c23b0abad396c181dbf38c56fcca559d88b9622af46bafd'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py', 'lf_sha256': '7e60c1cecfc8fa7809f7c836fa6fef1366806bd23c08e2890508ea26a711acac'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py', 'lf_sha256': '63c8f30bda20300d98d3d9a8a6478fc1a841cbaa3b334ee71688711e06375d97'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/loss.py', 'lf_sha256': '4aa2818948ebfd9fe416b3b310a053384508ef92c3b67831833fece560523084'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/sample_model.py', 'lf_sha256': '03fd683acc6d49d9b53e634b42bd150a3259a1a15c7dc8fe2badf115870a345b'}], 'checkpoint': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', 'checkpoint_sha256': 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52', 'parent_class': 'class QueryCompetitionIndicator(nn.Module):\n    """Predict confidence from a query and its overlapping competitors."""\n\n    def __init__(self, d_model):\n        super().__init__()\n        self.query_norm = nn.LayerNorm(d_model)\n        self.evidence_norm = nn.LayerNorm(3 * d_model + 1)\n        self.head = nn.Sequential(nn.Linear(3 * d_model + 1, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))\n\n    def forward(self, query, pred_masks, batch_mask, point_mass):\n        valid = (~batch_mask).unsqueeze(1).to(pred_masks.dtype)\n        probabilities = pred_masks.sigmoid() * valid\n        weighted = probabilities * point_mass.unsqueeze(1)\n        areas = weighted.sum(-1)\n        intersection = torch.einsum(\'bim,bjm->bij\', weighted, probabilities)\n        union = areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection\n        overlap = intersection / union.clamp_min(1e-06)\n        diagonal = torch.eye(overlap.shape[-1], dtype=torch.bool, device=overlap.device).unsqueeze(0)\n        overlap = overlap.masked_fill(diagonal, 0.0)\n        competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-06)\n        normalized_query = self.query_norm(query)\n        context = torch.einsum(\'bij,bjd->bid\', competition, normalized_query)\n        max_overlap = overlap.max(-1).values.unsqueeze(-1)\n        evidence = torch.cat([normalized_query, context, normalized_query - context, max_overlap], dim=-1)\n        return self.head(self.evidence_norm(evidence))', 'bn_eps': 0.0001, 'bn_momentum': 0.1}
"""Source-bound normalization mixture and fixed-input trained-head controls."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from torch import nn
import torch.nn.functional as F

assert P['physical_gpu'] is None
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=Path(P['root'])
assert (root/'.codex_source_commit').read_text().strip()==PAYLOAD['commit']
for row in PAYLOAD['sources']:
    assert hashlib.sha256((root/row['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==row['lf_sha256']
torch.set_num_threads(2)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


assert digest(PAYLOAD['checkpoint'])==PAYLOAD['checkpoint_sha256']
checkpoint=torch.load(PAYLOAD['checkpoint'],map_location='cpu')
weights=checkpoint['model']
affine={name:weights['output_layer.0.'+name].detach().clone() for name in
        ('weight','bias','running_mean','running_var','num_batches_tracked')}
head_weights={k.removeprefix('dec.indi_embedding.'):v for k,v in weights.items()
              if k.startswith('dec.indi_embedding.')}
del checkpoint,weights
scope=dict(torch=torch,nn=nn)
exec(PAYLOAD['parent_class'],scope)
Indicator=scope['QueryCompetitionIndicator']
rows=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999)
    width=int(affine['weight'].numel())
    bn=nn.BatchNorm1d(width,eps=PAYLOAD['bn_eps'],momentum=PAYLOAD['bn_momentum']).to(dtype=dtype)
    bn.load_state_dict({k:v.to(dtype=dtype) if v.is_floating_point() else v for k,v in affine.items()},strict=True)
    initial={k:v.clone() for k,v in bn.state_dict().items()}
    focal=torch.randn(32,width,dtype=dtype)
    mate_a=torch.randn(48,width,dtype=dtype)
    mate_b=3+2*torch.randn(48,width,dtype=dtype)
    values={}
    for mode in ('train','fixed_statistics'):
        for name,mate in (('A',mate_a),('B',mate_b),('repeatA',mate_a)):
            bn.load_state_dict(initial,strict=True)
            bn.train(mode=='train')
            values[mode,name]=bn(torch.cat((focal,mate),dim=0))[:len(focal)].detach()
    train_delta=float((values['train','A']-values['train','B']).abs().max())
    repeat=float((values['train','A']-values['train','repeatA']).abs().max())
    eval_delta=float((values['fixed_statistics','A']-values['fixed_statistics','B']).abs().max())
    bound=128*torch.finfo(dtype).eps*max(1.0,float(values['train','A'].abs().max()),float(values['train','B'].abs().max()))
    assert train_delta>bound and repeat==0 and eval_delta==0
    separated=[]
    for mate in (mate_a,mate_b):
        bn.load_state_dict(initial,strict=True)
        bn.train()
        separated.append(torch.cat((bn(focal),bn(mate)),dim=0)[:len(focal)].detach())
    assert torch.equal(*separated)
    permutation=torch.roll(torch.arange(len(focal)),3)
    bn.load_state_dict(initial,strict=True)
    bn.train()
    permuted=bn(torch.cat((focal[permutation],mate_a),dim=0))[:len(focal)][torch.argsort(permutation)].detach()
    permutation_error=float((values['train','A']-permuted).abs().max())
    assert permutation_error<=bound
    bn.load_state_dict(initial,strict=True)
    assert all(torch.equal(value,initial[key]) for key,value in bn.state_dict().items())
    head=Indicator(256).to(dtype=dtype)
    head.load_state_dict(head_weights,strict=True)
    head.eval()
    q=torch.randn(2,128,256,dtype=dtype)
    masks=torch.randn(2,128,17,dtype=dtype)
    padding=torch.zeros(2,17,dtype=torch.bool)
    padding[1,-3:]=True
    mass=torch.arange(1,18,dtype=dtype).repeat(2,1)
    with torch.no_grad():
        batch=head(q,masks,padding,mass)
        single=head(q[:1],masks[:1],padding[:1],mass[:1])
        altered_q=q.clone()
        altered_masks=masks.clone()
        altered_q[1]*=3
        altered_masks[1]+=2
        changed=head(altered_q,altered_masks,padding,mass)
        again=head(q,masks,padding,mass)
    head_bound=128*torch.finfo(dtype).eps*max(1.0,float(batch.abs().max()))
    head_batch_error=float((batch[0]-single[0]).abs().max())
    mate_head_error=float((batch[0]-changed[0]).abs().max())
    assert head_batch_error<=head_bound and mate_head_error<=head_bound and torch.equal(batch,again)
    rows.append(dict(dtype=str(dtype),actual_trained_affine='output_layer.0',channels=width,
        focal_rows=32,mate_rows=48,component_focal_mate_change=train_delta,component_repeat_error=repeat,
        fixed_stat_mate_error=eval_delta,separate_scene_mate_error=0,within_scene_permutation_error=permutation_error,
        arithmetic_bound=bound,BN_buffers_restored=True,
        actual_trained_head_batch_separability_error=head_batch_error,
        actual_trained_head_mate_change_error=mate_head_error,head_arithmetic_bound=head_bound,
        actual_trained_head_repeat_error=0))
assert not torch.cuda.is_initialized()
assert digest(PAYLOAD['checkpoint'])==PAYLOAD['checkpoint_sha256']
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y58',rows=rows,
    source_BN_config=dict(eps=PAYLOAD['bn_eps'],momentum=PAYLOAD['bn_momentum'],norm_eval=False),
    actual_checkpoint_affine_and_indicator=True,native_activations=False,native_loss=False,
    CUDA_initialized=False,optimizer_steps=0,weights_saved=False,admission=False,
    limitation='Synthetic feature component witness using actual trained affine; fixed-input real head negative control. '
               'Not actual sparse activation/task drift, native benefit/harm, prevalence, or a normalization replacement.')),
    encoding='utf-8')
