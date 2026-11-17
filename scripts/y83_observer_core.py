"""Exact private-query-normalization callbacks for CPU and native task checks."""
def norm_parameters():
    assert len(norms)==6 and len({p.data_ptr() for n in norms for p in n.parameters()})==12
    assert norms[0] is model.dec.norm_qqa and all(n.eps==norms[0].eps for n in norms)
    return [p for n in norms for p in n.parameters()]


def check_norm_rows(rows):
    assert len(rows)==12
    for i,r in enumerate(rows):
        assert type(r['step']) is int and r['step']==i//6 and type(r['stage']) is int and r['stage']==i%6
        for key in ('original_self_identity','original_language_identity','sum_input_exact','both_inputs_undetached','private_module_identity','original_epsilon','float64_LayerNorm_reference'):
            assert r[key] is True


def check_path_calls():
    assert len(state['qla'])==len(state['self'])==len(state['norm_values'])==6
    assert state['ffn_consumers']==list(range(6)) and state['coarse_calls']==6
    assert len(state['queries'])==7 and state['indicators']==list(range(7)) and state['query_norms']==[True]*7 and len(state['gates'])==7


def checked_prediction(q,features,pad):
    if not state['queries']:state['features'],state['padding']=features,pad
    assert features is state['features'] and pad is state['padding']
    state['queries'].append(q);value=original_prediction(q,features,pad);state['mask']=value[1]
    expected=(value[1].sigmoid()<.5)|pad.unsqueeze(1)
    expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(value[2],expected) and not value[2].requires_grad
    state['gates'].append(value[2]);return value


def capture_qla(module,args,value):
    i=len(state['qla']);assert module is model.dec.qla_layers[i]
    value[0].retain_grad();state['qla'].append(value[0])


def capture_self(module,args,value):
    i=len(state['self']);assert module is model.dec.qqa_layers[i] and args[0] is state['qla'][i]
    value.retain_grad();state['self'].append(value)


def capture_fusion_norm(module,args,value):
    i=len(state['norm_values']);assert module is norms[i] and len(args)==1
    assert torch.equal(args[0],state['self'][i]+state['qla'][i]) and args[0].requires_grad and value.requires_grad
    args[0].retain_grad();value.retain_grad()
    reference=torch.nn.functional.layer_norm(args[0].double(),module.normalized_shape,module.weight.double(),module.bias.double(),module.eps)
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    assert module.eps==model.dec.norm_qqa.eps
    state['sum_inputs'].append(args[0]);state['norm_values'].append(value)
    path_rows.append(dict(step=len(step_rows),stage=i,original_self_identity=True,original_language_identity=True,
        sum_input_exact=True,both_inputs_undetached=True,private_module_identity=True,original_epsilon=True,float64_LayerNorm_reference=True))


def before_ffn(module,args):
    i=len(state['ffn_consumers']);assert module is model.dec.qqa_ffn_layers[i] and args[0] is state['norm_values'][i]
    state['ffn_consumers'].append(i)


def before_coarse(module,args):
    i=state['coarse_calls'];assert module is model.dec.qia_layers[i] and args[3] is state['gates'][-1]
    state['coarse_calls']+=1


def before_indicator(module,args):
    assert args[0] is state['queries'][-1] and args[1] is state['mask'] and args[2] is state['padding']
    state['indicators'].append(len(state['queries'])-1)


def capture_query_norm(module,args,value):
    assert args[0] is state['queries'][-1] and args[0].requires_grad;state['query_norms'].append(True)


def reset_state():
    state.clear();state.update(features=None,padding=None,queries=[],gates=[],qla=[],self=[],norm_values=[],sum_inputs=[],ffn_consumers=[],coarse_calls=0,indicators=[],query_norms=[])


def gradient_record():
    check_path_calls();parameters=norm_parameters()
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in parameters)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['qla']+state['self']+state['sum_inputs']+state['norm_values'])
    heads=[p.grad for p in head.parameters() if p.grad is not None];assert len(heads)==10 and all(torch.isfinite(g).all() for g in heads)
    return dict(all_finite=True,tensors=len(active),indicator_tensors=10,indicator_calls=7,private_norm_calls=6,
        ffn_consumption_calls=6,original_coarse_gate_consumption_calls=6,query_norm_consumption_calls=7,
        norm_gradient_tensors=12,norm_gradient_maxima=[float(p.grad.abs().max()) for p in parameters],both_input_gradients_finite=True)


def install_hooks():
    hooks=[head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm)]
    hooks.extend(n.register_forward_hook(capture_fusion_norm) for n in norms)
    hooks.extend(n.register_forward_hook(capture_qla) for n in model.dec.qla_layers)
    hooks.extend(n.register_forward_hook(capture_self) for n in model.dec.qqa_layers)
    hooks.extend(n.register_forward_pre_hook(before_ffn) for n in model.dec.qqa_ffn_layers)
    hooks.extend(n.register_forward_pre_hook(before_coarse) for n in model.dec.qia_layers)
    return hooks
