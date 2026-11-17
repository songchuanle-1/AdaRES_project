"""Two original native TRAIN steps with current query-set to visual feedback observation."""
import os
assert P['physical_gpu'] == FROZEN['runtime_physical_gpu']
assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(P['physical_gpu'])
import subprocess
assert subprocess.check_output(['hostname']).decode().strip() == 'dl01-MS-7C60'
assert float(subprocess.check_output(['nvidia-smi', '-i', str(P['physical_gpu']), '--query-gpu=power.limit',
    '--format=csv,noheader,nounits']).decode().strip()) == FROZEN['runtime_power_limit_watts']
import base64
import hashlib
import importlib
import importlib.util
import itertools
import json
from functools import partial
from pathlib import Path
import sys

import gorilla
import torch
from torch import nn

assert torch.cuda.device_count() == 1
torch.cuda.set_device(0)
torch.set_num_threads(2)
stage = Path(P['output']).with_suffix('')
assert not stage.exists()
stage.mkdir()
native = importlib.import_module('train')
assert hashlib.sha256(Path(native.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == FROZEN['train_source_lf_sha256']
row = FROZEN['variant']
assert row['id'] == 'AdaRES-Y102'
package = stage / row['package']
package.mkdir()
for name, value in row['python_files'].items():
    raw = base64.b64decode(value['base64'])
    assert hashlib.sha256(raw).hexdigest() == value['sha256']
    (package / name).write_bytes(raw)
config_path = stage / Path(row['config']).name
raw = base64.b64decode(row['config_data']['base64'])
assert hashlib.sha256(raw).hexdigest() == row['config_data']['sha256']
config_path.write_bytes(raw)
cfg = gorilla.Config.fromfile(str(config_path))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30
assert cfg.dataloader.train.batch_size == 2
assert cfg.model.sampling_module.num_proposal == 128
assert cfg.model.dec.num_layer == 6 and cfg.model.dec.iter_pred
assert cfg.data.train.training and cfg.data.train.prefix == 'train'
cfg.dist = False
namespace = 'ipdn.model_paper5.' + row['package']
importlib.import_module('ipdn.model_paper5')
assert namespace not in sys.modules
spec = importlib.util.spec_from_file_location(namespace, str(package / '__init__.py'),
    submodule_search_locations=[str(package)])
module = importlib.util.module_from_spec(spec)
sys.modules[namespace] = module
spec.loader.exec_module(module)
Candidate = native.resolve_model_class(str(config_path))
assert Candidate is module.IPDN

gorilla.set_random_seed(cfg.train.seed)
model = Candidate(**cfg.model).cuda()
assert sum(p.numel() for p in model.parameters()) == 164630405
assert not hasattr(model.dec, 'refinement_indicators')
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)

dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
loader = native.build_dataloader(dataset, dist=False, **cfg.dataloader.train)






losses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}
head=model.dec.indi_embedding;residual=model.dec.geometry_mask_skip;original_prediction=model.dec.prediction_head
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

"""Execute only the first two original TRAIN batches after an authorized idle slot."""
class FirstTwoBatches:
    sampler = loader.sampler

    def __len__(self):
        return 2

    def __iter__(self):
        for batch in itertools.islice(loader, 2):
            assert len(batch['scan_ids']) == 2
            batches.append(dict(scan_ids=[str(scene) for scene in batch['scan_ids']],
                expressions=len(batch['ann_ids']), points=int(batch['feats'].shape[0])))
            yield batch


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def before_model(module, args):
    norms = [layer for block in (module.input_conv, module.unet, module.output_layer)
             for layer in block.modules() if isinstance(layer, nn.BatchNorm1d)]
    assert module.training and norms and all(layer.training for layer in norms)
    assert module.text_encoder.training and all(not parameter.requires_grad for parameter in module.text_encoder.parameters())
    modes.append(True)


def after_model(module, args, value):
    loss = value[0]
    assert loss.requires_grad and torch.isfinite(loss).all()
    losses.append(float(loss.detach().cpu()))


def before_decoder(module, args):
    assert len(args) == 8 and len(gradient_checks) == len(step_rows) and len(step_rows) < 2
    reset_state()
    state['raw_spatial_expected'] = args[0]
    state['raw_appearance_expected'] = args[6]
    state['raw_coordinates_expected'] = args[5]
    state['raw_mass_expected'] = args[7]
    state['batch_offsets_expected'] = args[2]


original_step = optimizer.step


def checked_step(*args, **kwargs):
    index = len(step_rows)
    assert index < 2 and len(losses) == len(modes) == index + 1
    record = gradient_record()
    before = residual.weight.detach().clone()
    value = original_step(*args, **kwargs)
    update = float((residual.weight.detach() - before).abs().max())
    assert update > 0 and torch.isfinite(residual.weight).all()
    record.update(step=index, parameter_update_max=update)
    gradient_checks.append(record)
    step_rows.append(dict(record))
    return value


hooks = install_hooks() + [model.register_forward_pre_hook(before_model), model.register_forward_hook(after_model),
                          model.dec.register_forward_pre_hook(before_decoder)]
model.dec.prediction_head = checked_prediction
optimizer.step = checked_step
torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1, model, FirstTwoBatches(), optimizer, scheduler, cfg, logger, Writer())
    torch.cuda.synchronize()
    assert len(losses) == len(gradient_checks) == len(batches) == len(modes) == 2
    assert len({scene for batch in batches for scene in batch['scan_ids']}) == 4
    check_response_rows(path_rows)
    peak = int(torch.cuda.max_memory_reserved())
    assert 0 < peak < 24 * 2**30
finally:
    optimizer.step = original_step
    model.dec.prediction_head = original_prediction
    for hook in hooks:
        hook.remove()
    if getattr(loader, '_iterator', None) is not None:
        loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(
    complete=True, id=row['id'], config=row['config'], code_commit=FROZEN['code_commit'],
    goal_sha256=FROZEN['goal_sha256'], native_YAML_sha256=row['config_data']['sha256'],
    native_entry='tools/train.py::train', split='train', steps=2, optimizer_steps=2, losses=losses,
    native_data_loaded=True, backward_finite=True, intended_path_observed=True, batches=batches,
    gradient_checks=gradient_checks, geometry_mask_paths=path_rows, geometry_mask_calls=2,
    original_visual_projection_and_mask_calls=2, coarse_gate_consumers=12, query_norm_consumption_calls=14, shared_head_calls=14,
    original_base_mask_and_static_reuse_preserved=True, original_hard_gate_algorithm_preserved=True, native_modes_verified=True,
    peak_reserved_bytes=peak, data_source=dict(dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
        config_data_root=cfg.data.train.data_root),
    validation_used=False, model_weights_saved=False, admission=False,
    limitation='Actual original native two-step TRAIN only; not trained efficacy or a full-dataset memory guarantee.')),
    encoding='utf8')
