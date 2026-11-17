"""Build original model/TRAIN CPU checks and frozen native callback body for Y81."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
core=(ROOT/'scripts/y81_observer_core.py').read_text(encoding='utf-8')
cpu=(ROOT/'scripts/probe_y80_cpu.py').read_text(encoding='utf-8')
cpu=cpu[:cpu.index('model.dec=decoder;optimizer=')]
cpu=cpu.replace('Y80','Y81').replace('y80','y81').replace('QueryVisualFeedback','QueryMaskRefinement').replace('query_visual_feedback','query_mask_refinement')
state="state.clear();state.update(features=None,padding=None,queries=[],values=[],mask_inputs=[],refinement_queries=[],gates=[],prediction_consumers=[],coarse_calls=0,indicators=[],norms=[])"
guards='''def check_path_calls(queries,values,predictions,coarse,indicators,norms):
    assert type(queries) is int and type(values) is int and type(coarse) is int and (queries,values,coarse)==(7,6,6)
    assert predictions==list(range(1,7)) and all(type(v) is int for v in predictions)
    assert indicators==list(range(7)) and all(type(v) is int for v in indicators)
    assert norms==[True]*7 and all(type(v) is bool for v in norms)

def check_response_rows(rows):
    from math import isfinite
    assert len(rows)==12
    for index,row in enumerate(rows):
        assert type(row['step']) is int and row['step']==index//6 and type(row['stage']) is int and row['stage']==index%6+1
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero']==bool(index>=6)
        for key in ('original_static_mask_base_identity','current_refined_query_identity','original_padding_identity','both_inputs_undetached','padded_output_identity','float64_attention_reference_checked'):
            assert row[key] is True
        delta=row['mask_feature_delta_max'];assert isfinite(delta) and delta>=0
        if index<6:assert delta==0
        else:assert delta>0

'''
hooks='''hooks=[refinement.register_forward_hook(capture_refinement),head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm)]
hooks.extend(layer.register_forward_pre_hook(before_coarse) for layer in model.dec.qia_layers)
model.dec.prediction_head=checked_prediction
'''
cpu+='''model.dec=decoder;optimizer=gorilla.build_optimizer(model,cfg.optimizer)
step_rows=[];path_rows=[];state={};original_prediction=model.dec.prediction_head
head=model.dec.indi_embedding;refinement=model.dec.query_mask_refinement
'''+guards+core+hooks+'''try:
    model.train(True)
    for index in range(2):
        '''+state+'''
        optimizer.zero_grad();torch.manual_seed(1999+index);value=model.dec(**inputs)
        check_path_calls(len(state['queries']),len(state['values']),state['prediction_consumers'],state['coarse_calls'],state['indicators'],state['norms'])
        labels=torch.arange(256).reshape(2,128).remainder(2).long()
        loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts(value))
        assert torch.isfinite(loss);loss.backward()
        active=[p.grad for p in model.parameters() if p.grad is not None];assert active and all(torch.isfinite(g).all() for g in active)
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['mask_inputs']+state['refinement_queries']+state['values'])
        grad=refinement.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
        headgrads=[p.grad for p in head.parameters() if p.grad is not None];assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
        before=refinement.weight.detach().clone();optimizer.step();update=float((refinement.weight.detach()-before).abs().max());assert update>0
        step_rows.append(dict(step=index,synthetic_CPU_decoder_loss=float(loss),finite_gradient_tensors=len(active),original_head_gradient_tensors=10,
            shared_head_calls=7,mask_refinement_calls=6,prediction_consumption_calls=6,original_coarse_gate_consumption_calls=6,
            both_input_gradients_finite=True,new_gradient_max=float(grad.abs().max()),parameter_update_max=update))
    check_response_rows(path_rows)
finally:
    model.dec.prediction_head=original_prediction
    for hook in hooks:hook.remove()
model.eval();assert all(not m.training for m in model.modules());model.train(True);assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],implementation_test_passed=True,
    full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),parameters=parameters,additional_parameters=65536,component_controls=controls,
    decoder_initial_RNG_and_parent_weights_exact=True,decoder_all_stages_anchor_exact=True,decoder_input_gradients_anchor_close=True,original_initial_prediction_unchanged=True,
    exact_static_mask_current_query_padding_identities_verified=True,both_input_paths_undetached=True,padded_output_identity=True,
    original_prediction_and_hard_gate_consumption_verified=True,original_shared_head_query_norm_consumption_verified=True,
    synthetic_CPU_decoder_optimizer_steps=2,CPU_decoder_steps=step_rows,CPU_refinement_paths=path_rows,
    nonzero_refinement_changes_refined_and_final_masks=True,bypassed_refinement_negative_control_exact=True,
    CUDA_initialized=False,native_smoke_passed=False,native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='CPU full native model/pretrain/TRAIN collation and synthetic component/decoder optimizer checks only; no native sparse TRAIN task/GPU PASS/admission/efficacy.')),encoding='utf8')
'''
native=(ROOT/'scripts/probe_y80_native_smoke.py').read_text(encoding='utf-8').split('losses=[]',1)[0].replace('Y80','Y81').replace('y80','y81')
native+='''losses=[];gradient_checks=[];batches=[];modes=[];step_rows=[];path_rows=[];state={}
head=model.dec.indi_embedding;refinement=model.dec.query_mask_refinement
assert len(list(head.parameters()))==10 and torch.equal(refinement.weight,torch.zeros_like(refinement.weight))
original_prediction=model.dec.prediction_head
'''+guards+'''class FirstTwoBatches:
    sampler=loader.sampler
    def __len__(self):return 2
    def __iter__(self):
        for batch in itertools.islice(loader,2):
            assert len(batch['scan_ids'])==2
            batches.append(dict(scan_ids=[str(s) for s in batch['scan_ids']],expressions=len(batch['ann_ids']),points=int(batch['feats'].shape[0])))
            yield batch
class Writer:
    def add_scalar(self,*args,**kwargs):pass

def before_model(module,args):
    bn=[m for block in (module.input_conv,module.unet,module.output_layer) for m in block.modules() if isinstance(m,nn.BatchNorm1d)]
    assert module.training and bn and all(m.training for m in bn)
    assert module.text_encoder.training and all(not p.requires_grad for p in module.text_encoder.parameters());modes.append(True)

def after_model(module,args,value):
    loss=value[0];assert loss.requires_grad and torch.isfinite(loss).all();losses.append(float(loss.detach().cpu()))

def before_decoder(module,args):
    assert len(args)==8 and len(gradient_checks)==len(step_rows) and len(step_rows)<2
    '''+state+'\n\n'+core+'''step=optimizer.step

def checked_step(*args,**kwargs):
    index=len(gradient_checks);assert index<2 and len(losses)==len(modes)==index+1
    check_path_calls(len(state['queries']),len(state['values']),state['prediction_consumers'],state['coarse_calls'],state['indicators'],state['norms'])
    active=[p.grad for p in model.parameters() if p.grad is not None];assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    headgrads=[p.grad for p in head.parameters() if p.grad is not None];assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['mask_inputs']+state['refinement_queries']+state['values'])
    grad=refinement.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
    before=refinement.weight.detach().clone();gradient_max=float(grad.abs().max());result=step(*args,**kwargs)
    update=float((refinement.weight.detach()-before).abs().max());assert update>0 and torch.isfinite(refinement.weight).all()
    record=dict(all_finite=True,tensors=len(active),indicator_tensors=10,indicator_calls=7,mask_refinement_calls=6,prediction_consumption_calls=6,
        original_coarse_gate_consumption_calls=6,query_norm_consumption_calls=7,both_input_gradients_finite=True,refinement_weight_gradient_max=gradient_max,parameter_update_max=update)
    gradient_checks.append(record);step_rows.append(dict(record));return result

'''+hooks+'''hooks.extend([model.register_forward_pre_hook(before_model),model.register_forward_hook(after_model),model.dec.register_forward_pre_hook(before_decoder)])
optimizer.step=checked_step;torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer());torch.cuda.synchronize()
    assert len(losses)==len(gradient_checks)==len(batches)==len(modes)==2 and len({scene for batch in batches for scene in batch['scan_ids']})==4
    check_response_rows(path_rows);peak=int(torch.cuda.max_memory_reserved());assert 0<peak<24*2**30
finally:
    optimizer.step=step;model.dec.prediction_head=original_prediction
    for hook in hooks:hook.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',steps=2,optimizer_steps=2,losses=losses,
    native_data_loaded=True,backward_finite=True,intended_path_observed=True,batches=batches,gradient_checks=gradient_checks,refinement_paths=path_rows,
    mask_refinement_calls=12,prediction_consumption_calls=12,original_coarse_gate_consumption_calls=12,query_norm_consumption_calls=14,shared_head_calls=14,
    original_static_mask_base_preserved=True,original_hard_attention_algorithm_preserved=True,original_initial_prediction_preserved=True,
    native_modes_verified=True,peak_reserved_bytes=peak,data_source=dict(dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,config_data_root=cfg.data.train.data_root),
    validation_used=False,model_weights_saved=False,admission=False,limitation='Actual native two-step TRAIN only; no efficacy or full-dataset memory guarantee.')),encoding='utf8')
'''
for name,source in [('probe_y81_cpu.py',cpu),('probe_y81_native_smoke.py',native)]:
    path=ROOT/'scripts'/name;assert not path.exists();ast.parse(source,feature_version=(3,8));path.write_text(source,encoding='utf-8',newline='\n')
print('Created Y81 CPU/native probe sources with exact shared observer functions; no model/task execution')
