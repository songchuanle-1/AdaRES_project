"""Pending task-facing frozen-mode diagnosis, preserving original controls."""

import gc
import inspect

assert digest(Path(inspect.getfile(type(dataset))))==SCOPE['dataset_source']['sha256']
items=[]
for scene in FROZEN['scenes'][:2]:
    gorilla.set_random_seed(1999)
    item=dataset[scene['dataset_index']]
    single=dataset.collate_fn([copy.deepcopy(item)])
    assert str(single['scan_ids'][0])==scene['scan_id']
    assert [str(v) for v in single['ann_ids']]==scene['ann_ids']
    items.append(item)
batch=dataset.collate_fn(copy.deepcopy(items))
for name,parameter in model.named_parameters():
    parameter.requires_grad_(not name.startswith('text_encoder.'))
rows=[]
focal_logits={}
sampling={}
for role in ('original','original_repeat','restored_frozen_mode','restored_repeat'):
    for name,value in model.named_buffers():value.copy_(buffers[name])
    model.train()
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.dropout._DropoutNd) and not name.startswith('text_encoder.'):
            module.eval()
    if role.startswith('restored'):
        model.text_encoder.eval()
    model.zero_grad(set_to_none=True)
    gorilla.set_random_seed(1999)
    outputs=[]
    rng_state={}
    decoded=[]
    def before_encoder(module,args):
        rng_state['cpu']=torch.get_rng_state()
        rng_state['cuda']=torch.cuda.get_rng_state_all()
    def after_encoder(module,args,value):
        # Dropout must not shift the random stream used by downstream operators.
        torch.set_rng_state(rng_state['cpu'])
        torch.cuda.set_rng_state_all(rng_state['cuda'])
    def observe_decoder(module,args,value):
        decoded.append(value['sample_inds'].detach().cpu().clone())
    def observe(module,args,value):
        value.retain_grad()
        outputs.append(value)
    hooks=[model.dec.indi_embedding.register_forward_hook(observe),
           model.text_encoder.register_forward_pre_hook(before_encoder),
           model.text_encoder.register_forward_hook(after_encoder),
           model.dec.register_forward_hook(observe_decoder)]
    torch.cuda.reset_peak_memory_stats()
    try:
        loss,logs=model(copy.deepcopy(batch),mode='loss')
        assert torch.isfinite(loss).all() and len(outputs)==7 and len(decoded)==1
        loss.backward()
        gradients={}
        for name,parameter in model.dec.indi_embedding.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            gradients[name]=float(parameter.grad.abs().max())
            assert gradients[name]>0
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in outputs)
        active=[p.grad for p in model.parameters() if p.grad is not None]
        assert active and all(torch.isfinite(g).all() for g in active)
        assert all(p.grad is None and not p.requires_grad for p in model.text_encoder.parameters())
        torch.cuda.synchronize()
        peak=int(torch.cuda.max_memory_reserved())
        assert peak<24*2**30
        focal_logits[role]=outputs[-1][:4].detach().cpu().clone()
        sampling[role]=decoded[0][:4]
        rows.append(dict(role=role,native_loss=float(loss.item()),indicator_gradients=gradients,
            all7_native_task_logits_finite=True,encoder_training=model.text_encoder.training,
            active_encoder_dropout=sum(isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.training
                                       for m in model.text_encoder.modules()),
            encoder_parameters_frozen=True,peak_reserved_bytes=peak))
    finally:
        for hook in hooks:hook.remove()
    outputs.clear()
    model.zero_grad(set_to_none=True)
    del loss,logs,active
    gc.collect()
comparisons={}
for first,second,label in (('original','original_repeat','original_repeat'),
                           ('restored_frozen_mode','restored_repeat','restored_repeat'),
                           ('original','restored_frozen_mode','mode_intervention')):
    a,b=focal_logits[first],focal_logits[second]
    comparisons[label]=dict(logit_max=float((a-b).abs().max()),
        confidence_slot_max=float((a.softmax(-1)[...,1]-b.softmax(-1)[...,1]).abs().max()),
        same_focal_sampling_indices=torch.equal(sampling[first],sampling[second]),
        interpretation='Slot-paired only if learned sampling differs; not fixed-query confidence.')
for name,value in model.named_buffers():value.copy_(buffers[name])
assert parameter_versions=={name:value._version for name,value in model.named_parameters()}
assert digest(checkpoint_path)==FROZEN['checkpoint_sha256']
assert (root/'.codex_source_commit').read_text().strip()==FROZEN['commit']
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y59',rows=rows,cases=comparisons,
    actual_sparse_TRAIN_loss_backward=True,trained_W2_diagnostic_only=True,
    exact_same_batch_and_seed=True,other_dropout_held_disabled=True,BN_mode_unchanged=True,
    encoder_RNG_consumption_isolated=True,
    model_weights_unchanged=True,buffers_restored=True,optimizer_steps=0,weights_saved=False,
    admission=False,limitation='Native mode-contract mechanism diagnosis only. No independent candidate '
        'package, released-init candidate gate, optimizer test, resource Pareto or trained benefit/harm.')),
    encoding='utf-8')
