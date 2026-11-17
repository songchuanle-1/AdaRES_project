"""Physical anchored Y57 native controls/backward and bounded paired resources."""

import statistics
from ipdn.model_paper5.adares_w2_pointmass_competition.ipdn import IPDN as Parent

model=initialize()
dataset,items,native_batches=batches()
initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
pretrain_sha=digest(cfg.train.pretrain)
rows=[]
for index,batch in enumerate(native_batches):
    model.load_state_dict(initial,strict=True)
    model.train()
    model.zero_grad(set_to_none=True)
    gorilla.set_random_seed(1999)
    evidence,inputs,anchors=[],[],[]
    def observe_evidence(module,args):
        args[0].retain_grad()
        evidence.append(args[0])
    def observe_head(module,args,value):
        inputs.append(tuple(v.detach().clone() for v in args))
        if isinstance(value,tuple):
            value[1].retain_grad()
            anchors.append(value[1])
    handles=[model.dec.indi_embedding.evidence_norm.register_forward_pre_hook(observe_evidence),
             model.dec.indi_embedding.register_forward_hook(observe_head)]
    torch.cuda.reset_peak_memory_stats()
    try:
        loss,logs=model(copy.deepcopy(batch),mode='loss')
        loss.backward()
        assert torch.isfinite(loss).all() and len(evidence)==len(inputs)==7 and len(anchors)==1
        gradients={}
        for name,p in model.dec.indi_embedding.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().max())>0
            gradients[name]=float(p.grad.abs().max())
        assert all(e.grad is not None and torch.isfinite(e.grad).all() and float(e.grad.abs().max())>0 for e in evidence)
        assert all(torch.equal(e.detach()[...,-256:],anchors[0].detach()) for e in evidence)
        assert anchors[0].grad is not None and torch.isfinite(anchors[0].grad).all()
        anchor_gradient=float(anchors[0].grad.abs().max())
        assert anchor_gradient>0
        reserved=int(torch.cuda.max_memory_reserved())
        assert reserved<24*2**30
    finally:
        for h in handles:h.remove()
    anchor=anchors[0].detach()
    head=model.dec.indi_embedding
    controls=[]
    with torch.no_grad():
        for layer,args in enumerate(inputs):
            q,m,padding,mass=args
            baseline=head(q,m,padding,mass,initial_peer_variance=anchor)
            repeat=head(q,m,padding,mass,initial_peer_variance=anchor)
            perm=torch.roll(torch.arange(128,device=q.device),1)
            permuted=head(q[:,perm],m[:,perm],padding,mass,initial_peer_variance=anchor[:,perm])[:,torch.argsort(perm)]
            shifted=head(q,m,padding,mass,initial_peer_variance=torch.roll(anchor,1,dims=1))
            bound=128*torch.finfo(q.dtype).eps*max(1.0,float(baseline.abs().max()))
            assert torch.equal(baseline,repeat)
            error=float((baseline-permuted).abs().max())
            assert error<=bound
            signal=float((baseline-shifted).abs().max())
            controls.append(dict(layer=layer,repeat_error=0,permutation_error=error,
                                 original_arithmetic_bound=bound,matched_anchor_shift_signal=signal))
        assert controls[6]['matched_anchor_shift_signal']>controls[6]['original_arithmetic_bound']
    rows.append(dict(batch=index,loss=float(loss.item()),indicator_gradients=gradients,
        all7_anchor_paths_reached=True,initial_anchor_gradient_max=anchor_gradient,
        controls=controls,peak_reserved_bytes=reserved,finite_backward=True))
    evidence.clear()
    anchors.clear()
    inputs.clear()
    model.zero_grad(set_to_none=True)
    del loss,logs,anchor,q,m,baseline,repeat,permuted,shifted
    gc.collect()

model.load_state_dict(initial,strict=True)
points=[int(b['coords_float'].shape[0]) for b in native_batches]
largest=max(range(4),key=lambda i:points[i])
stress=[]
for iteration in range(5):
    model.load_state_dict(initial,strict=True)
    model.train()
    model.zero_grad(set_to_none=True)
    gorilla.set_random_seed(1999)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    loss,logs=model(copy.deepcopy(native_batches[largest]),mode='loss')
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss).all() and torch.cuda.max_memory_reserved()<24*2**30
    stress.append(dict(iteration=iteration,warmup=iteration<2,seconds=time.perf_counter()-start,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),finite_backward=True))
    del loss,logs
    model.zero_grad(set_to_none=True)
    gc.collect()
model.load_state_dict(initial,strict=True)
model.cpu()
torch.cuda.empty_cache()
parent=Parent(**cfg.model)
common={k:v for k,v in initial.items() if k in parent.state_dict() and parent.state_dict()[k].shape==v.shape}
loaded=parent.load_state_dict(common,strict=False)
assert sorted(loaded.missing_keys)==['dec.indi_embedding.evidence_norm.bias','dec.indi_embedding.evidence_norm.weight','dec.indi_embedding.head.0.weight']
counts=dict(parent=sum(p.numel() for p in parent.parameters()),candidate=sum(p.numel() for p in model.parameters()))
assert counts['candidate']-counts['parent']==66048
profile=[]
for role,tested in (('parent',parent),('candidate',model)):
    tested.cuda()
    tested.eval()
    with torch.no_grad():
        for index,item in enumerate(items):
            batch=dataset.collate_fn([copy.deepcopy(item)])
            times=[]
            for iteration in range(5):
                gorilla.set_random_seed(1999)
                current=copy.deepcopy(batch)
                torch.cuda.synchronize()
                start=time.perf_counter()
                value=tested(current,mode='predict')
                torch.cuda.synchronize()
                assert all(torch.as_tensor(v).numel()==128 for v in value['indi'])
                if iteration>=2:times.append(time.perf_counter()-start)
                del current,value
            profile.append(dict(role=role,scene=PAYLOAD['scenes'][index]['scan_id'],
                                seconds=times,median_seconds=statistics.median(times)))
    tested.cpu()
    gc.collect()
    torch.cuda.empty_cache()
assert digest(cfg.train.pretrain)==pretrain_sha and preserved()
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y57',revision=1,rows=rows,
    physical_revised_package=True,actual_native_loss_backward=True,original_final_control_bound_unchanged=True,
    parameter_counts=counts,parameter_delta=66048,stress=stress,paired_native_profile=profile,
    same_common_initial_weights=True,optimizer_steps=0,weights_saved=False,admission=False,
    limitation='Independent physical revised connection controls/native/resources only; original failure retained, trained efficacy unknown.')),
    encoding='utf-8')
