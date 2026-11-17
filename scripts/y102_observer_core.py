"""Shared exact original geometry/static-mask consumers for CPU and native checks."""
def reset_state():
    state.clear();state.update(predictions=[],masks=[],gates=[],indicators=[],norms=[],coarse=0,mass=None)


def capture_projection(module,args,value):
    key='spatial_projected' if module is model.dec.input_proj else 'appearance_projected'
    assert key not in state and value.requires_grad
    expected=state['raw_spatial_expected'] if key=='spatial_projected' else state['raw_appearance_expected']
    assert args[0] is expected
    value.retain_grad();state[key]=value


def capture_base(module,args,value):
    assert module is model.dec.x_mask and args[0] is state['raw_spatial_expected'] and value.requires_grad
    value.retain_grad();state['mask_base_unbatched']=value


def capture_skip(module,args,value):
    assert module is residual and len(args)==4 and 'mask_adapted' not in state
    base,pos,padding,mass=args
    expected,ep=model.dec.get_batches(state['mask_base_unbatched'],state['batch_offsets_expected'])
    coords,_=model.dec.get_batches(state['raw_coordinates_expected'],state['batch_offsets_expected'])
    masses,_=model.dec.get_batches(state['raw_mass_expected'].unsqueeze(-1),state['batch_offsets_expected'])
    assert torch.equal(base,expected) and torch.equal(padding,ep) and torch.equal(pos,coords) and torch.equal(mass,masses.squeeze(-1))
    assert pos.shape[-1]==3 and base.shape[-1]==256 and base.requires_grad and value.requires_grad
    clean=pos.double().masked_fill(padding.unsqueeze(-1),0.)
    weights=mass.double().masked_fill(padding,0.);totals=weights.sum(1,keepdim=True)
    center=(clean*weights.unsqueeze(-1)).sum(1)/totals.clamp_min(1.)
    relative=(clean-center.unsqueeze(1)).masked_fill(padding.unsqueeze(-1)|(totals<=0).unsqueeze(-1),0.)
    reference=base.double()+relative@module.weight.double()
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    assert torch.equal(value[padding],base[padding])
    base.retain_grad();value.retain_grad();state.update(mask_base=base,mask_adapted=value,padding=padding,original_mass=mass)
    nonzero=bool(torch.count_nonzero(module.weight));delta=float((value-base).abs().max())
    assert nonzero==bool(len(step_rows)) and (delta>0 if nonzero else delta==0)
    path_rows.append(dict(step=len(step_rows),weight_nonzero=nonzero,mask_feature_delta_max=delta,
        original_full3_coordinates_and_point_masses=True,original_unbatched_mask_and_batching=True,
        original_spatial_and_appearance_projections_executed=True,float64_all_valid_scene_centroid_reference=True,
        original_padding_rows_preserved=True,original_base_undetached=True,all_seven_predictions_reuse_same_features=False))


def checked_prediction(query,features,padding):
    assert features is state['mask_adapted'] and padding is state['padding']
    state['predictions'].append(query);value=original_prediction(query,features,padding);state['masks'].append(value[1])
    expected=(value[1].sigmoid()<.5)|padding.unsqueeze(1);expected[expected.all(-1)]=False;expected=expected|padding.unsqueeze(1)
    assert torch.equal(value[2],expected) and not value[2].requires_grad
    state['gates'].append(value[2])
    if len(state['predictions'])==7:path_rows[-1]['all_seven_predictions_reuse_same_features']=True
    return value


def before_coarse(module,args):
    index=state['coarse'];assert module is model.dec.qia_layers[index] and args[3] is state['gates'][-1];state['coarse']+=1


def before_indicator(module,args):
    assert module is head and args[0] is state['predictions'][-1] and args[1] is state['masks'][-1]
    assert args[2] is state['padding'] and args[3] is state['original_mass'] and not args[3].requires_grad
    state['indicators'].append(len(state['predictions'])-1)


def capture_query_norm(module,args,value):
    assert module is head.query_norm and args[0] is state['predictions'][-1] and args[0].requires_grad and value.requires_grad
    state['norms'].append(True)


def gradient_record():
    assert len(state['predictions'])==len(state['masks'])==len(state['gates'])==7
    assert state['indicators']==list(range(7)) and state['norms']==[True]*7 and state['coarse']==6
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    hg=[p.grad for p in head.parameters() if p.grad is not None];assert len(hg)==10 and all(torch.isfinite(g).all() for g in hg)
    original=list(model.dec.x_mask.parameters())+list(model.dec.input_proj.parameters())+list(model.dec.input_proj_2d.parameters())
    assert len(original)==14 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in original)
    for k in ('mask_base_unbatched','mask_base','mask_adapted','spatial_projected','appearance_projected'):
        assert state[k].grad is not None and torch.isfinite(state[k].grad).all()
    g=residual.weight.grad;assert g is not None and torch.isfinite(g).all() and g.abs().max()>0
    return dict(all_finite=True,tensors=len(active),indicator_tensors=10,indicator_calls=7,prediction_calls=7,query_norm_consumption_calls=7,
        coarse_gate_consumers=6,geometry_mask_calls=1,original_visual_parameter_tensors=14,
        original_base_adapted_and_visual_gradients_finite=True,frozen_text_encoder_preserved=True,new_gradient_max=float(g.abs().max()))


def check_response_rows(rows):
    import math
    assert len(rows)==2
    for i,r in enumerate(rows):
        assert type(r['step']) is int and r['step']==i and type(r['weight_nonzero']) is bool and r['weight_nonzero']==bool(i)
        for k in ('original_full3_coordinates_and_point_masses','original_unbatched_mask_and_batching',
                  'original_spatial_and_appearance_projections_executed','float64_all_valid_scene_centroid_reference',
                  'original_padding_rows_preserved','original_base_undetached','all_seven_predictions_reuse_same_features'):
            assert r[k] is True,k
        assert math.isfinite(r['mask_feature_delta_max']) and (r['mask_feature_delta_max']>0 if i else r['mask_feature_delta_max']==0)


def install_hooks():
    hooks=[model.dec.input_proj.register_forward_hook(capture_projection),model.dec.input_proj_2d.register_forward_hook(capture_projection),
        model.dec.x_mask.register_forward_hook(capture_base),residual.register_forward_hook(capture_skip),
        head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm)]
    hooks.extend(m.register_forward_pre_hook(before_coarse) for m in model.dec.qia_layers)
    return hooks
