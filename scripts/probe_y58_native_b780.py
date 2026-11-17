"""Pending exclusive native batch-mate/BN diagnosis; never a training launch."""

import gc
import inspect

assert digest(Path(inspect.getfile(type(dataset))))==SCOPE['dataset_source']['sha256']
items=[]
for scene in FROZEN['scenes'][:3]:
    gorilla.set_random_seed(1999)
    item=dataset[scene['dataset_index']]
    single=dataset.collate_fn([copy.deepcopy(item)])
    assert str(single['scan_ids'][0])==scene['scan_id']
    assert [str(v) for v in single['ann_ids']]==scene['ann_ids']
    items.append(item)
single=dataset.collate_fn([copy.deepcopy(items[0])])
focal_points=int(single['coords_float'].shape[0])
focal_expressions=len(single['ann_ids'])
captures={}
rows=[]
for mode in ('pooled_training_BN','fixed_running_BN'):
    for label,mate in (('A',1),('B',2),('repeatA',1)):
        for name,value in model.named_buffers():
            value.copy_(buffers[name])
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.dropout._DropoutNd):
                module.eval()
            if mode=='fixed_running_BN' and isinstance(module,torch.nn.BatchNorm1d):
                module.eval()
        batch=dataset.collate_fn(copy.deepcopy([items[0],items[mate]]))
        assert torch.equal(batch['coords_float'][:focal_points],single['coords_float'])
        assert torch.equal(batch['feats'][:focal_points],single['feats'])
        assert [str(v) for v in batch['ann_ids'][:focal_expressions]]==FROZEN['scenes'][0]['ann_ids']
        activation=[]
        outputs=[]
        def observe_backbone(module,args,value):
            chosen=value.indices[:,0]==0
            coords=value.indices[chosen].detach().cpu().numpy()
            features=value.features[chosen].detach().cpu()
            order=np.lexsort((coords[:,3],coords[:,2],coords[:,1]))
            activation.append((coords[order].copy(),features[torch.from_numpy(order).long()].clone()))
        def observe_decoder(module,args,value):
            seeds=value.get('sample_inds')
            seeds=seeds[:focal_expressions].detach().cpu().clone() if torch.is_tensor(seeds) else None
            outputs.append((value['indis'][:focal_expressions].detach().cpu().clone(),seeds))
        hooks=[model.output_layer.register_forward_hook(observe_backbone),model.dec.register_forward_hook(observe_decoder)]
        gorilla.set_random_seed(1999)
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                loss,_=model(copy.deepcopy(batch),mode='loss')
            assert torch.isfinite(loss).all() and len(activation)==len(outputs)==1
            assert torch.isfinite(activation[0][1]).all() and torch.isfinite(outputs[0][0]).all()
            torch.cuda.synchronize()
            peak=int(torch.cuda.max_memory_reserved())
            assert peak<24*2**30
            captures[mode,label]=dict(coords=activation[0][0],features=activation[0][1],
                                     logits=outputs[0][0],seeds=outputs[0][1])
            rows.append(dict(mode=mode,mate=label,mate_scene=FROZEN['scenes'][mate]['scan_id'],
                total_native_loss=float(loss.item()),focal_voxels=len(activation[0][0]),
                peak_reserved_bytes=peak,dropout_modules_training=sum(
                    isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.training for m in model.modules()),
                BN_training_modules=sum(isinstance(m,torch.nn.BatchNorm1d) and m.training for m in model.modules())))
        finally:
            for hook in hooks:hook.remove()
        del loss,batch
        gc.collect()
comparisons=[]
for mode in ('pooled_training_BN','fixed_running_BN'):
    a,b,repeat=(captures[mode,label] for label in ('A','B','repeatA'))
    assert np.array_equal(a['coords'],b['coords']) and np.array_equal(a['coords'],repeat['coords'])
    same_seeds=(torch.equal(a['seeds'],b['seeds']) if a['seeds'] is not None and b['seeds'] is not None else None)
    delta=a['features']-b['features']
    comparisons.append(dict(mode=mode,focal_voxel_coordinates_identical=True,
        feature_mate_max=float(delta.abs().max()),feature_mate_RMS=float(delta.square().mean().sqrt()),
        feature_repeat_max=float((a['features']-repeat['features']).abs().max()),
        same_focal_sampling_indices=same_seeds,
        confidence_slot_mate_max=float((a['logits'].softmax(-1)[...,1]-b['logits'].softmax(-1)[...,1]).abs().max()),
        confidence_slot_repeat_max=float((a['logits'].softmax(-1)[...,1]-repeat['logits'].softmax(-1)[...,1]).abs().max()),
        confidence_interpretation='Slot-paired only; not a matched fixed-query confidence contrast if sampling changed.',
        total_loss_interpretation='Includes a different mate target; not a signed task benefit/harm contrast.'))
for name,value in model.named_buffers():value.copy_(buffers[name])
assert parameter_versions=={name:value._version for name,value in model.named_parameters()}
assert all(torch.equal(value,buffers[name]) for name,value in model.named_buffers())
assert digest(checkpoint_path)==FROZEN['checkpoint_sha256']
assert (root/'.codex_source_commit').read_text().strip()==FROZEN['commit']
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y58',rows=rows,cases=comparisons,
    actual_sparse_model=True,actual_TRAIN_batch2=True,model_weights_unchanged=True,buffers_restored=True,
    dropout_held_disabled_in_all_arms=True,optimizer_steps=0,weights_saved=False,admission=False,
    limitation='Native trained-parent context diagnosis only, not released-init candidate backward, '
               'population harm/benefit or permission to replace normalization.')),
    encoding='utf-8')
