"""Six exact callbacks actually executed by the recorded Y84 CPU decoder check."""
def tracked_batch(*args,**kwargs):
    result=original_batch(*args,**kwargs);state['batches'].append(result);return result

def capture_generated(module,args,value):
    assert 'generated' not in state;state['generated']=value

def capture_initial(module,args,value):
    assert len(args)==4 and len(state['batches'])==4 and 'initialized' not in state
    q,f,pad,mass=args;assert q is state['generated'] and f is state['batches'][0][0] and pad is state['batches'][0][1]
    assert mass.data_ptr()==state['batches'][2][0].data_ptr() and torch.equal(mass,state['batches'][2][0].squeeze(-1))
    assert q.requires_grad and f.requires_grad and value.requires_grad;q.retain_grad();f.retain_grad();value.retain_grad()
    weights=mass.masked_fill(pad,0).double();clean=f.double().masked_fill(pad.unsqueeze(-1),0)
    totals=weights.sum(1,keepdim=True);context=(clean*weights.unsqueeze(-1)).sum(1)/totals.clamp_min(1)
    context=context.masked_fill(totals<=0,0);reference=q.double()+(context@module.weight.double()).unsqueeze(1)
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    state['initialized']=value;state['scene']=f
    nonzero=bool(torch.count_nonzero(module.weight));delta=float((value-q).abs().max())
    assert nonzero==bool(len(step_rows)) and (delta>0 if nonzero else delta==0)
    path_rows.append(dict(step=len(step_rows),original_generated_query_identity=True,original_scene_visual_identity=True,original_padding_and_point_mass_identity=True,all_inputs_undetached=True,float64_weighted_context_reference=True,weight_nonzero=nonzero,query_delta_max=delta))

def checked_prediction(q,features,pad):
    assert features is state['batches'][1][0] and pad is state['batches'][0][1]
    if state['predictions']==0:assert q is state['initialized']
    state['query']=q;state['predictions']+=1;value=original_prediction(q,features,pad);state['mask']=value[1]
    expected=(value[1].sigmoid()<.5)|pad.unsqueeze(1);expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(value[2],expected) and not value[2].requires_grad;return value

def before_indicator(module,args):
    assert args[0] is state['query'] and args[1] is state['mask'];state['indicators']+=1

def capture_norm(module,args,value):
    assert args[0] is state['query'];state['norms']+=1
