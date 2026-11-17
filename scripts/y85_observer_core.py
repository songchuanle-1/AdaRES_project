"""Seven exact callbacks actually executed during the recorded Y85 CPU check."""
def capture_mask_raw(module,args,value):
    assert 'raw' not in state and value.requires_grad;value.retain_grad();state['raw']=value

def capture_mask_norm(module,args,value):
    assert len(args)==1 and args[0] is state['raw'] and value.requires_grad and 'normalized' not in state
    assert module is norm and module.eps==1e-5 and tuple(module.normalized_shape)==(256,)
    reference=torch.nn.functional.layer_norm(args[0].double(),module.normalized_shape,module.weight.double(),module.bias.double(),module.eps)
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    value.retain_grad();state['normalized']=value
    path_rows.append(dict(step=len(step_rows),original_x_mask_output_identity=True,all_channels_and_original_epsilon=True,input_undetached=True,float64_LayerNorm_reference=True,shared_module_identity=True))

def tracked_batch(*args,**kwargs):
    if len(state['batches'])==1:assert args[0] is state['normalized']
    result=original_batch(*args,**kwargs);state['batches'].append(result);return result

def checked_prediction(q,features,pad):
    assert len(state['batches'])==4 and features is state['batches'][1][0] and pad is state['batches'][0][1]
    assert torch.count_nonzero(features[pad])==0
    state['query']=q;state['predictions']+=1;value=original_prediction(q,features,pad);state['mask']=value[1]
    expected=(value[1].sigmoid()<.5)|pad.unsqueeze(1);expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(value[2],expected) and not value[2].requires_grad;state['gate']=value[2];return value

def before_indicator(module,args):
    assert args[0] is state['query'] and args[1] is state['mask'];state['indicators']+=1

def capture_query_norm(module,args,value):
    assert args[0] is state['query'] and args[0].requires_grad;state['query_norms']+=1

def before_coarse(module,args):
    assert module is model.dec.qia_layers[state['coarse_calls']] and args[3] is state['gate'];state['coarse_calls']+=1
