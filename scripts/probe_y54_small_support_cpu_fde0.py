"""Adversarial small-support controls on the exact independent decoder, CPU only."""
import base64
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

assert P['physical_gpu'] is None and os.environ['CUDA_VISIBLE_DEVICES']==''
import gorilla
import torch
assert torch.cuda.device_count()==0
torch.set_num_threads(2)
root=Path(P['root'])
for row in PAYLOAD['parent_files']:
    assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
directory=Path(tempfile.mkdtemp(prefix='adares_y54_small_support_'+P['token']+'_'))
package=directory/'adares_y54_fine_support'
package.mkdir()
for name,row in PAYLOAD['files'].items():
    data=base64.b64decode(row['base64'])
    assert hashlib.sha256(data).hexdigest()==row['sha256']
    (package/name if name.endswith('.py') else directory/name).write_bytes(data)
cfg=gorilla.Config.fromfile(str(directory/'adares_y54_fine_support_res.yaml'))
namespace='ipdn.model_paper5.adares_y54_fine_support'
importlib.import_module('ipdn.model_paper5')
assert namespace not in sys.modules
spec=importlib.util.spec_from_file_location(namespace,str(package/'__init__.py'),submodule_search_locations=[str(package)])
module=importlib.util.module_from_spec(spec)
sys.modules[namespace]=module
spec.loader.exec_module(module)
CandidateDEC=importlib.import_module(namespace+'.dec').DEC
ParentDEC=importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec').DEC
gorilla.set_random_seed(1999)
candidate=CandidateDEC(**cfg.model.dec,in_channel=cfg.model.media,sampling_module=cfg.model.sampling_module).cpu().eval()
parent=ParentDEC(**cfg.model.dec,in_channel=cfg.model.media,sampling_module=cfg.model.sampling_module).cpu().eval()
parent.load_state_dict(candidate.state_dict(),strict=True)
for value in (candidate,parent):
    for parameter in value.parameters():
        parameter.requires_grad_(False)
rows=[]
started=time.perf_counter()
for small in (1,8,128,406,512):
    gorilla.set_random_seed(1999)
    counts=[small,512]
    total=sum(counts)
    x=torch.randn(total,cfg.model.media,requires_grad=True)
    positions=torch.randn(total,3)
    clip=torch.randn(total,1024)
    masses=torch.randint(1,100,(total,)).float()
    offsets=torch.tensor([0,small,total],dtype=torch.int32)
    fps=torch.stack((torch.arange(256)%small,torch.arange(256)))
    language=torch.randn(2,78,768)
    valid=torch.zeros(2,78,dtype=torch.long)
    valid[0,:7]=1
    valid[1,:13]=1
    observed=[]
    originals=[layer.forward for layer in candidate.qia_layers_fine]

    def observer(original):
        def forward(source,query,batch_mask=None,attn_mask=None,pe=None,att_weights=None):
            assert batch_mask is not None
            assert ((~batch_mask.unsqueeze(1)) & (~attn_mask)).any(-1).all()
            source.retain_grad()
            observed.append((source,batch_mask.detach()))
            return original(source,query,batch_mask,attn_mask,pe,att_weights)
        return forward

    try:
        for layer,original in zip(candidate.qia_layers_fine,originals):
            layer.forward=observer(original)
        out=candidate(x,fps,offsets,language,valid,positions,clip,masses)
        layers=[out]+out['aux_outputs']
        assert len(layers)==7 and all(torch.isfinite(value['indis']).all() and torch.isfinite(value['masks']).all() for value in layers)
        objective=sum(value['indis'].square().sum() for value in layers)
        objective.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert len(observed)==6
        maximum=0.0
        for source,invalid in observed:
            assert source.grad is not None and torch.isfinite(source.grad).all()
            if invalid.any():
                delta=float(source.grad[invalid].abs().max())
                assert delta==0
                maximum=max(maximum,delta)
        equal_parent=None
        if small==512:
            with torch.no_grad():
                old=parent(x.detach(),fps,offsets,language,valid,positions,clip,masses)
            equal_parent=max(float((a['indis']-b['indis']).abs().max()) for a,b in zip(layers,[old]+old['aux_outputs']))
            assert equal_parent==0
        rows.append(dict(valid_supports=counts,actual_native_decoder=True,component_fixtures_not_actual_FPS=True,all6neighborhoods_nonempty=True,all7layer_outputs_finite=True,backward_finite=True,padded_support_gradient_max_abs=maximum,all_valid_parent_indicator_max_abs=equal_parent))
    finally:
        for layer,original in zip(candidate.qia_layers_fine,originals):
            layer.forward=original
    observed.clear()
    del out,layers,objective,x
    gc.collect()
result=dict(complete=True,id='AdaRES-Y54',actual_independent_package=True,cpu_only=True,case_count=5,rows=rows,elapsed_seconds=time.perf_counter()-started,model_or_config_changes=False,admission=False,limitation='Adversarial component fixtures on exact native decoder, not actual sparse/FPS examples or trained efficacy. All cases keep fixed native512fine/128query settings; no hyperparameter search or weight persistence.')
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
