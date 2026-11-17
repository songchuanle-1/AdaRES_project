"""Same-ID preregistered initial-moment connection diagnostic, not admission."""

model=initialize()
dataset,items,native_batches=batches()
initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
rows=[]
for index,batch in enumerate(native_batches):
    for role in ('original','initial_moment_anchor','anchor_repeat'):
        model.load_state_dict(initial,strict=True)
        model.train()
        model.zero_grad(set_to_none=True)
        gorilla.set_random_seed(1999)
        anchors=[]
        outputs=[]
        layer_inputs=[]
        def connect(module,args):
            value=args[0]
            if not anchors:
                anchor=value[...,-256:]
                anchor.retain_grad()
                anchors.append(anchor)
            changed=value if role=='original' else torch.cat((value[...,:769],anchors[0]),dim=-1)
            layer_inputs.append(changed.detach().clone())
            return (changed,)
        def observe(module,args,value):
            value.retain_grad()
            outputs.append(value)
        hooks=[model.dec.indi_embedding.evidence_norm.register_forward_pre_hook(connect),
               model.dec.indi_embedding.register_forward_hook(observe)]
        try:
            loss,logs=model(copy.deepcopy(batch),mode='loss')
            loss.backward()
            assert torch.isfinite(loss).all() and len(outputs)==7
            gradients={n:float(p.grad.abs().max()) for n,p in model.dec.indi_embedding.named_parameters()
                       if p.grad is not None and torch.isfinite(p.grad).all()}
            assert len(gradients)==len(list(model.dec.indi_embedding.named_parameters()))
            assert all(v>0 for v in gradients.values())
            assert all(o.grad is not None and torch.isfinite(o.grad).all() and float(o.grad.abs().max())>0 for o in outputs)
            if role!='original':
                assert anchors[0].grad is not None and torch.isfinite(anchors[0].grad).all()
                assert float(anchors[0].grad.abs().max())>0
            final=layer_inputs[6]
            shifted=final.clone()
            shifted[...,-256:]=torch.roll(shifted[...,-256:],1,dims=1)
            head=model.dec.indi_embedding
            with torch.no_grad():
                actual=head.head(head.evidence_norm(final))
                changed=head.head(head.evidence_norm(shifted))
            bound=128*torch.finfo(final.dtype).eps*max(1.0,float(actual.abs().max()))
            rows.append(dict(batch=index,role=role,loss=float(loss.item()),
                final_fixed_shift_signal=float((actual-changed).abs().max()),original_arithmetic_bound=bound,
                original_control_passed=float((actual-changed).abs().max())>bound,
                indicator_gradients=gradients,all7_native_logit_gradients=True,
                initial_anchor_gradient_max=None if anchors[0].grad is None else float(anchors[0].grad.abs().max()),
                initial_variance_max=float(layer_inputs[0][...,-256:].abs().max()),
                final_variance_max=float(final[...,-256:].abs().max())))
        finally:
            for h in hooks:h.remove()
        outputs.clear()
        anchors.clear()
        layer_inputs.clear()
        del loss,logs,final,shifted,actual,changed
        model.zero_grad(set_to_none=True)
        gc.collect()
model.load_state_dict(initial,strict=True)
assert preserved()
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y57',revision=1,rows=rows,
    physical_original_package=True,connection_intervention_only=True,
    old_failed_controls_retained=True,control_bounds_unchanged=True,
    optimizer_steps=0,weights_saved=False,admission=False,
    limitation='Preregistered local connection diagnostic only; independent physical realization/resources still pending.')),
    encoding='utf-8')
