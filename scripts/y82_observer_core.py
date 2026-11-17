"""Callbacks shared literally by CPU decoder and future native TRAIN checks."""
def check_path_calls(state):
    for key in ('coarse','fine','fused','coarse_inputs','fine_inputs'):
        assert len(state[key])==6
    assert len(state['queries'])==7 and len(state['gates'])==7
    assert state['qla_consumers']==list(range(6)) and all(type(i) is int for i in state['qla_consumers'])
    assert state['indicators']==list(range(7)) and state['norms']==[True]*7


def check_response_rows(rows):
    from math import isfinite
    assert len(rows)==12
    for index,row in enumerate(rows):
        assert type(row['step']) is int and row['step']==index//6
        assert type(row['stage']) is int and row['stage']==index%6
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero']==bool(index>=6)
        for key in ('original_coarse_identity','original_fine_identity','both_inputs_undetached','float64_product_projection_reference'):
            assert row[key] is True
        delta=row['fusion_delta_max'];assert isfinite(delta) and delta>=0
        assert delta==0 if index<6 else delta>0


def checked_prediction(q,features,pad):
    if not state['queries']:state['features'],state['padding']=features,pad
    assert features is state['features'] and pad is state['padding']
    state['queries'].append(q);value=original_prediction(q,features,pad);state['mask']=value[1]
    expected=(value[1].sigmoid()<.5)|pad.unsqueeze(1)
    expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(value[2],expected) and not value[2].requires_grad
    state['gates'].append(value[2]);return value


def capture_coarse(module,args,value):
    index=len(state['coarse'])
    assert module is model.dec.qia_layers[index] and args[3] is state['gates'][-1]
    state['coarse'].append(value[0])


def capture_fine(module,args,value):
    index=len(state['fine']);assert module is model.dec.qia_layers_fine[index]
    state['fine'].append(value[0])


def capture_fusion(module,args,value):
    index=len(state['fused']);assert len(args)==2 and args[0] is state['coarse'][index] and args[1] is state['fine'][index]
    assert all(t.requires_grad for t in args)
    for t in args+(value,):t.retain_grad()
    reference=(args[0].double()+args[1].double())/2+(args[0].double()*args[1].double())@module.weight.double()
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    nonzero=bool(module.weight.abs().max()>0);assert nonzero==bool(len(step_rows))
    delta=float((value-(args[0]+args[1])/2).abs().max());assert (delta>0)==nonzero
    state['coarse_inputs'].append(args[0]);state['fine_inputs'].append(args[1]);state['fused'].append(value)
    path_rows.append(dict(step=len(step_rows),stage=index,weight_nonzero=nonzero,fusion_delta_max=delta,
        original_coarse_identity=True,original_fine_identity=True,both_inputs_undetached=True,float64_product_projection_reference=True))


def before_qla(module,args):
    index=len(state['qla_consumers']);assert module is model.dec.qla_layers[index] and args[1] is state['fused'][index]
    state['qla_consumers'].append(index)


def before_indicator(module,args):
    assert args[0] is state['queries'][-1] and args[1] is state['mask'] and args[2] is state['padding']
    state['indicators'].append(len(state['queries'])-1)


def capture_query_norm(module,args,value):
    assert args[0] is state['queries'][-1] and args[0].requires_grad;state['norms'].append(True)


def reset_state():
    state.clear();state.update(features=None,padding=None,queries=[],gates=[],coarse=[],fine=[],fused=[],coarse_inputs=[],fine_inputs=[],qla_consumers=[],indicators=[],norms=[])


def gradient_record():
    check_path_calls(state)
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    headgrads=[p.grad for p in head.parameters() if p.grad is not None]
    assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['coarse_inputs']+state['fine_inputs']+state['fused'])
    grad=fusion.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
    return dict(all_finite=True,tensors=len(active),indicator_tensors=10,indicator_calls=7,fusion_calls=6,
        qla_consumption_calls=6,query_norm_consumption_calls=7,both_input_gradients_finite=True,new_gradient_max=float(grad.abs().max()))


def install_hooks():
    hooks=[fusion.register_forward_hook(capture_fusion),head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm)]
    hooks.extend(m.register_forward_hook(capture_coarse) for m in model.dec.qia_layers)
    hooks.extend(m.register_forward_hook(capture_fine) for m in model.dec.qia_layers_fine)
    hooks.extend(m.register_forward_pre_hook(before_qla) for m in model.dec.qla_layers)
    return hooks
