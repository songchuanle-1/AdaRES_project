"""Independent scalar all-scene centered-coordinate projection controls."""
parameters=sum(p.numel() for p in model.parameters());assert parameters==164630405
parent_path=root/'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==PAYLOAD['parent_dec_lf_sha256']
parent_module=importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module=importlib.import_module(namespace+'.dec');controls=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999);rng=torch.get_rng_state().clone();layer=dec_module.GeometryMaskSkip(7).to(dtype=dtype)
    assert torch.equal(rng,torch.get_rng_state())
    base=torch.randn(2,5,7,dtype=dtype,requires_grad=True);pos=torch.randn(2,5,3,dtype=dtype,requires_grad=True)
    mass=(torch.arange(10,dtype=dtype).reshape(2,5)+1).requires_grad_();padding=torch.tensor([[False,False,False,True,True],[False,False,False,False,True]])
    anchor=layer(base,pos,padding,mass);assert torch.equal(anchor,base)
    grads=torch.autograd.grad(anchor.sum(),(base,pos,mass,layer.weight))
    assert torch.equal(grads[0],torch.ones_like(base)) and torch.equal(grads[1],torch.zeros_like(pos)) and torch.equal(grads[2],torch.zeros_like(mass))
    assert torch.isfinite(grads[3]).all() and grads[3].abs().max()>0
    with torch.no_grad():
        layer.weight.copy_((torch.arange(21,dtype=dtype).reshape(3,7)-10)/21)
        value=layer(base,pos,padding,mass);reference=base.detach().double().clone()
        for b in range(2):
            total=sum(float(mass[b,j]) for j in range(5) if not padding[b,j])
            center=[sum(float(pos[b,j,a])*float(mass[b,j]) for j in range(5) if not padding[b,j])/total for a in range(3)]
            for j in range(5):
                if not padding[b,j]:
                    for d in range(7):reference[b,j,d]+=sum((float(pos[b,j,a])-center[a])*float(layer.weight[a,d]) for a in range(3))
        tol=128*torch.finfo(dtype).eps;torch.testing.assert_allclose(value.double(),reference,rtol=tol,atol=tol)
        order=torch.tensor([3,1,4,0,2]);torch.testing.assert_allclose(layer(base[:,order],pos[:,order],padding[:,order],mass[:,order]),value[:,order],rtol=tol,atol=tol)
        assert torch.equal(layer(base.flip(0),pos.flip(0),padding.flip(0),mass.flip(0)),value.flip(0))
        torch.testing.assert_allclose(layer(base,pos+3,padding,mass),value,rtol=tol,atol=tol)
        torch.testing.assert_allclose(layer(base,pos,padding,mass*2),value,rtol=tol,atol=tol)
        dirty=pos.detach().clone();dirty[padding]=float('nan');dm=mass.detach().clone();dm[padding]=float('nan')
        assert torch.equal(layer(base,dirty,padding,dm),value)
        assert torch.equal(layer(base,pos,torch.ones_like(padding),mass),base)
        assert torch.equal(layer(base,pos,padding,torch.zeros_like(mass)),base)
        moved=pos.detach().clone();moved[0,0,0]+=1;changed=layer(base,moved,padding,mass)
        assert torch.equal(changed[1],value[1]) and not torch.equal(changed[0],value[0])
        shifted=mass.detach().clone();shifted[0,0]+=2;assert not torch.allclose(layer(base,pos,padding,shifted),value)
        axis=torch.tensor([2,0,1]);other=dec_module.GeometryMaskSkip(7).to(dtype=dtype);other.weight.copy_(layer.weight[axis])
        torch.testing.assert_allclose(other(base,pos[:,:,axis],padding,mass),value,rtol=tol,atol=tol)
    grads=torch.autograd.grad(layer(base,pos,padding,mass).square().mean(),(base,pos,mass,layer.weight))
    assert all(torch.isfinite(g).all() and g.abs().max()>0 for g in grads)
    assert torch.count_nonzero(grads[1][padding])==torch.count_nonzero(grads[2][padding])==0
    with torch.no_grad():layer.weight.zero_();assert torch.equal(layer(base,pos,padding,mass),anchor)
    controls.append(dict(dtype=str(dtype),no_RNG=True,zero_identity_and_extra_input_gradients_exact=True,
        independent_scalar_all_scene_centroid_projection=True,scene_point_axis_permutations=True,
        coordinate_translation_and_positive_mass_scale_invariant=True,padded_NaN_positions_and_masses_ignored=True,
        empty_and_zero_mass_noop=True,scene_isolation_and_coordinate_mass_interventions=True,
        all_input_and_weight_gradients_finite_nonzero=True,padded_coordinate_and_mass_gradients_zero=True,exact_zero_restoration=True))
