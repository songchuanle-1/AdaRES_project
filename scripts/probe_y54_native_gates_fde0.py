"""Append to independent candidate setup: native gradients, restore, cost and predict."""
import gc
import random
import statistics
import numpy as np
from ipdn.model_paper5.adares_w2_pointmass_competition.ipdn import IPDN as Parent

exec(compile(PAYLOAD['state_helpers'], '<byte-bound-tested-state-helpers>', 'exec'), globals())
initial_state = cpu_copy(model.state_dict())
gorilla.set_random_seed(1999)
parent = Parent(**cfg.model).cpu()
parent.load_state_dict(initial_state, strict=True)
assert equal_state(initial_state, parent.state_dict())
parameter_counts = dict(parent=sum(p.numel() for p in parent.parameters()), candidate=sum(p.numel() for p in model.parameters()))
assert parameter_counts['parent'] == parameter_counts['candidate']
items = []
for scene in frozen:
    gorilla.set_random_seed(1999)
    item = dataset[scene['dataset_index']]
    single = dataset.collate_fn([copy.deepcopy(item)])
    assert single['scan_ids'][0] == scene['scan_id']
    assert [str(v) for v in single['ann_ids']] == [v['ann_id'] for v in scene['expressions']]
    items.append(item)
batches = [dataset.collate_fn(copy.deepcopy(items[i:i+2])) for i in range(0, 8, 2)]
points = [int(batch['coords_float'].shape[0]) for batch in batches]
largest = max(range(4), key=lambda i: points[i])
fine_sources = {}
original_fine = [layer.forward for layer in model.dec.qia_layers_fine]


def fine_observer(index, original):
    def forward(source, query, batch_mask=None, attn_mask=None, pe=None, att_weights=None):
        assert batch_mask is not None and batch_mask.dtype == torch.bool
        assert batch_mask.shape == source.shape[:2] and source.requires_grad
        assert ((~batch_mask.unsqueeze(1)) & (~attn_mask)).any(-1).all(), 'Corrected fine attention has an empty valid neighborhood'
        source.retain_grad()
        fine_sources[index] = (source, batch_mask.detach())
        return original(source, query, batch_mask, attn_mask, pe, att_weights)
    return forward


native_rows = []
torch.cuda.reset_peak_memory_stats()
try:
    for index, layer in enumerate(model.dec.qia_layers_fine):
        layer.forward = fine_observer(index, original_fine[index])
    for index, batch in enumerate(batches):
        model.load_state_dict(initial_state, strict=True)
        model.train()
        gorilla.set_random_seed(1999)
        model.zero_grad(set_to_none=True)
        fine_sources.clear()
        loss, logs = model(copy.deepcopy(batch), mode='loss')
        assert torch.isfinite(loss).all()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert len(fine_sources) == 6
        layers = []
        for layer_index in range(6):
            source, invalid = fine_sources[layer_index]
            assert source.grad is not None and torch.isfinite(source.grad).all()
            padding_gradient = float(source.grad[invalid].abs().max()) if invalid.any() else None
            if padding_gradient is not None:
                assert padding_gradient == 0
            valid_gradient = float(source.grad[~invalid].abs().max())
            assert valid_gradient > 0
            attention_gradients = [p.grad for p in model.dec.qia_layers_fine[layer_index].attn.parameters() if p.grad is not None]
            assert attention_gradients and all(torch.isfinite(g).all() for g in attention_gradients)
            assert any(float(g.abs().max()) > 0 for g in attention_gradients)
            layers.append(dict(layer=layer_index, padded_fine_supports=int(invalid.sum()), padding_source_gradient_max_abs=padding_gradient, valid_source_gradient_max_abs=valid_gradient, finite_attention_gradients=True))
        indicator = [p.grad for p in model.dec.indi_embedding.head.parameters() if p.grad is not None]
        assert indicator and all(torch.isfinite(g).all() for g in indicator) and any(float(g.abs().max())>0 for g in indicator)
        native_rows.append(dict(batch=index, expressions=int(sum(batch['scenes_len'])), native_loss=float(loss), finite_backward=True, gradient_tensors=len(gradients), layers=layers, finite_nonzero_indicator_gradients=True))
        print(json.dumps(dict(native_backward=native_rows[-1])), flush=True)
        fine_sources.clear()
        del loss, logs, gradients, source, attention_gradients, indicator
        model.zero_grad(set_to_none=True)
        gc.collect()
finally:
    for layer, original in zip(model.dec.qia_layers_fine, original_fine):
        layer.forward = original
    fine_sources.clear()
assert sum(row['expressions'] for row in native_rows) == 32

stress = []
for iteration in range(5):
    model.load_state_dict(initial_state, strict=True)
    model.train()
    gorilla.set_random_seed(1999)
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    loss, logs = model(copy.deepcopy(batches[largest]), mode='loss')
    assert torch.isfinite(loss).all()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    torch.cuda.synchronize()
    reserved = int(torch.cuda.max_memory_reserved())
    assert reserved < 24*2**30
    stress.append(dict(iteration=iteration, warmup=iteration<2, seconds=time.perf_counter()-begin, finite_backward=True, peak_reserved_bytes=reserved))
    del loss, logs, gradients
    model.zero_grad(set_to_none=True)
    gc.collect()

model.load_state_dict(initial_state, strict=True)
model.train()
gorilla.set_random_seed(1999)
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
steps = []
snapshot = None
torch.cuda.reset_peak_memory_stats()
for index in range(2):
    optimizer.zero_grad(set_to_none=True)
    loss, logs = model(copy.deepcopy(batches[index]), mode='loss')
    assert torch.isfinite(loss).all()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip))
    optimizer.step()
    scheduler.step()
    steps.append(dict(step=index+1, loss=float(loss), native_gradient_norm=norm, optimizer_state_entries=len(optimizer.state), scheduler_epoch=scheduler.last_epoch))
    del loss, logs, gradients
    if index == 0:
        snapshot = (cpu_copy(model.state_dict()), cpu_copy(optimizer.state_dict()), cpu_copy(scheduler.state_dict()), rng_state())
    gc.collect()
assert snapshot is not None and steps[0]['optimizer_state_entries'] > 0
assert not equal_state(snapshot[0], model.state_dict())
model.load_state_dict(snapshot[0], strict=True)
optimizer.load_state_dict(snapshot[1])
scheduler.load_state_dict(snapshot[2])
restore_rng(snapshot[3])
restored = dict(model=equal_state(snapshot[0], model.state_dict()), optimizer=equal_state(snapshot[1], optimizer.state_dict()), scheduler=equal_state(snapshot[2], scheduler.state_dict()), python_rng=random.getstate()==snapshot[3][0], numpy_rng=all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(np.random.get_state(),snapshot[3][1])), torch_rng=torch.equal(torch.get_rng_state(),snapshot[3][2]), cuda_rng=all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),snapshot[3][3])))
assert all(restored.values())
native_training_peak = max(int(torch.cuda.max_memory_reserved()), max(row['peak_reserved_bytes'] for row in stress))
assert native_training_peak < 24*2**30
model.zero_grad(set_to_none=True)
del optimizer, scheduler, snapshot, initial_state, batches
gc.collect()
torch.cuda.empty_cache()

checkpoint_path = Path('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth')
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == PAYLOAD['checkpoint_sha256']
checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
weights = dict(checkpoint['model'])
extras = set(weights)-set(parent.state_dict())
assert extras.issubset({'text_encoder.embeddings.position_ids','text_encoder.embeddings.token_type_ids'})
buffers = dict(parent.named_buffers())
for key in extras:
    assert key in buffers and torch.equal(weights[key].cpu(),buffers[key].cpu()) and not buffers[key].requires_grad
    del weights[key]
parent.load_state_dict(weights, strict=True)
model.load_state_dict(weights, strict=True)
del weights, checkpoint
model.cpu()
torch.cuda.empty_cache()
observations = {'parent': {}, 'candidate': {}}
profiles = []


def prediction_snapshot(value):
    return dict(identities=[(str(a),str(b)) for a,b in zip(value['scan_id'],value['ann_id'])], masks=[torch.as_tensor(v).detach().cpu().clone() for v in value['pred_pmask']], pious=[float(v) for v in value['piou']], confidence=[torch.as_tensor(v).detach().cpu().clone() for v in value['indi']], selected=[int(torch.as_tensor(v).argmax()) for v in value['indi']])


def prediction_difference(left,right):
    assert left['identities']==right['identities']
    return dict(probability_max_abs=max(float((a-b).abs().max()) for a,b in zip(left['masks'],right['masks'])), binary_mask_points_changed=sum(int(((a>.5)!=(b>.5)).sum()) for a,b in zip(left['masks'],right['masks'])), selected_query_changes=sum(a!=b for a,b in zip(left['selected'],right['selected'])), maximum_expression_iou_difference=max(abs(a-b) for a,b in zip(left['pious'],right['pious'])))


for role, tested in (('parent',parent),('candidate',model)):
    tested.cuda()
    tested.eval()
    for index,item in enumerate(items):
        batch = dataset.collate_fn([copy.deepcopy(item)])
        with torch.no_grad():
            for _ in range(2):
                gorilla.set_random_seed(1999)
                tested(copy.deepcopy(batch),mode='predict')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings, outputs = [], []
            for _ in range(3):
                gorilla.set_random_seed(1999)
                torch.cuda.synchronize()
                begin=time.perf_counter()
                value=tested(copy.deepcopy(batch),mode='predict')
                torch.cuda.synchronize()
                timings.append(time.perf_counter()-begin)
                outputs.append(prediction_snapshot(value))
                assert all(torch.as_tensor(v).numel()==128 for v in value['indi'])
                del value
        observations[role][index]=outputs
        profiles.append(dict(role=role,scene=frozen[index]['scan_id'],warmups=2,measured_seconds=timings,median_seconds=statistics.median(timings),peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),peak_reserved_bytes=int(torch.cuda.max_memory_reserved())))
    tested.cpu()
    gc.collect()
    torch.cuda.empty_cache()
parity=[]
for index in range(8):
    a,b=observations['parent'][index],observations['candidate'][index]
    parity.append(dict(scene=frozen[index]['scan_id'],matched=prediction_difference(a[0],b[0]),parent_repeat=[prediction_difference(a[0],v) for v in a[1:]],candidate_repeat=[prediction_difference(b[0],v) for v in b[1:]]))
y36_dir=Path('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y36_local_adjacency_res')
y36_paths=sorted(y36_dir.glob('*/best.pth'))
y36_binding=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size) for path in y36_paths]
result=dict(complete=True,id='AdaRES-Y54',actual_independent_package=True,parameter_counts=parameter_counts,parameter_delta=0,fixed_TRAIN_scenes=8,expressions=32,actual_native_sparse_FPS_backward=True,native_backward=native_rows,stress_input=dict(batch=largest,points=points[largest],selection='point-largest among original four frozen TRAIN batchsize2 pairs'),stress=stress,optimizer_steps=steps,nonempty_optimizer_restore=restored,native_training_peak_reserved_bytes=native_training_peak,inference_state='SAME fixed trained W2 diagnostic weights, NOT trained Y54',inference_profile=profiles,inference_parity=parity,champion_Y36_archive_intake=y36_binding,unique_existing_Y36_best=len(y36_binding)==1,model_or_checkpoint_weights_saved=False,admission=False,training_launches=0,limitation='Native fine-path finite/zero-padding gradients, RAM-only two-step exact restore and bounded fixed-input resource/predict controls. Not trained Y54 efficacy, global worst-case resource, universal numeric parity or Pareto superiority. No Git commit or admission.')
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
