"""Exact callbacks shared by synthetic CPU and actual native Y81 task checks."""
def checked_prediction(q,features,pad):
    index=len(state['queries'])
    if index==0:
        assert not state['values'];state['features']=features;state['padding']=pad
    else:
        assert q is state['refinement_queries'][-1] and features is state['values'][-1] and pad is state['padding']
        assert len(state['values'])==index;state['prediction_consumers'].append(index)
    state['queries'].append(q);result=original_prediction(q,features,pad);state['mask']=result[1]
    expected=(result[1].sigmoid()<.5)|pad.unsqueeze(1)
    expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(result[2],expected) and not result[2].requires_grad
    state['gates'].append(result[2]);return result


def capture_refinement(module,args,value):
    index=len(state['values'])
    assert len(args)==3 and args[0] is state['features'] and args[2] is state['padding'] and len(state['queries'])==index+1
    assert all(t.requires_grad for t in args[:2])
    args[0].retain_grad();args[1].retain_grad();value.retain_grad()
    attention=torch.softmax(args[0].double()@args[1].double().transpose(-1,-2)/(args[1].shape[-1]**.5),dim=-1)
    reference=args[0].double()+((attention@args[1].double())@module.weight.double()).masked_fill(args[2].unsqueeze(-1),0.)
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    nonzero=bool(module.weight.abs().max()>0);assert nonzero==bool(len(step_rows))
    delta=float((value-args[0]).abs().max());assert (delta>0)==nonzero and torch.equal(value[args[2]],args[0][args[2]])
    state['values'].append(value);state['mask_inputs'].append(args[0]);state['refinement_queries'].append(args[1])
    path_rows.append(dict(step=len(step_rows),stage=index+1,weight_nonzero=nonzero,mask_feature_delta_max=delta,
        original_static_mask_base_identity=True,current_refined_query_identity=True,original_padding_identity=True,
        both_inputs_undetached=True,padded_output_identity=True,float64_attention_reference_checked=True))


def before_coarse(module,args):
    index=state['coarse_calls'];assert module is model.dec.qia_layers[index] and args[3] is state['gates'][-1]
    state['coarse_calls']+=1


def before_indicator(module,args):
    assert args[0] is state['queries'][-1] and args[1] is state['mask'] and args[2] is state['padding']
    state['indicators'].append(len(state['queries'])-1)


def capture_query_norm(module,args,value):
    assert args[0] is state['queries'][-1] and args[0].requires_grad;state['norms'].append(True)
