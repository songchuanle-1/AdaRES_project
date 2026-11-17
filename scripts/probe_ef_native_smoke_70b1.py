"""Two real optimizer steps through native YAML resolution and native train()."""

import os
assert P['physical_gpu'] in (0,1)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(P['physical_gpu'])
import base64
import gc
import hashlib
import importlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import gorilla
import torch
from torch import nn
import yaml

assert torch.cuda.device_count()==1
torch.cuda.set_device(0)
torch.set_num_threads(2)
root=Path(P['root'])
stage=Path(P['output']).with_suffix('')
assert not stage.exists()
stage.mkdir()
native=importlib.import_module('train')
assert hashlib.sha256(Path(native.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==FROZEN['train_source_lf_sha256']
row=FROZEN['variant']
package=stage/row['package']
package.mkdir()
for name,value in row['python_files'].items():
    data=base64.b64decode(value['base64'])
    assert hashlib.sha256(data).hexdigest()==value['sha256']
    (package/name).write_bytes(data)
config_path=stage/Path(row['config']).name
config_data=base64.b64decode(row['config_data']['base64'])
assert hashlib.sha256(config_data).hexdigest()==row['config_data']['sha256']
config_path.write_bytes(config_data)
cfg=gorilla.Config.fromfile(str(config_path))
assert cfg.train.seed==1999 and cfg.train.epochs==30 and cfg.dataloader.train.batch_size==2
assert cfg.model.sampling_module.num_proposal==128
assert cfg.data.train.training and cfg.data.train.prefix=='train'
cfg.dist=False
namespace='ipdn.model_paper5.'+row['package']
importlib.import_module('ipdn.model_paper5')
assert namespace not in sys.modules
spec=importlib.util.spec_from_file_location(namespace,str(package/'__init__.py'),
    submodule_search_locations=[str(package)])
module=importlib.util.module_from_spec(spec)
sys.modules[namespace]=module
spec.loader.exec_module(module)
Candidate=native.resolve_model_class(str(config_path))
assert Candidate is module.IPDN
gorilla.set_random_seed(cfg.train.seed)
model=Candidate(**cfg.model).cuda()
optimizer=gorilla.build_optimizer(model,cfg.optimizer)
scheduler=gorilla.build_lr_scheduler(optimizer,cfg.lr_scheduler)
logger=importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model,cfg.train.pretrain,logger)
dataset=native.resolve_train_dataset(str(config_path),cfg.data.train.copy(),logger)
loader=native.build_dataloader(dataset,dist=False,**cfg.dataloader.train)
losses=[]
gradients=[]
batches=[]
path_checks=[]

class FirstTwoBatches:
    sampler=loader.sampler
    def __len__(self):return 2
    def __iter__(self):
        for batch in itertools.islice(loader,2):
            assert len(batch['scan_ids'])==cfg.dataloader.train.batch_size
            batches.append(dict(scan_ids=[str(s) for s in batch['scan_ids']],
                expressions=len(batch['ann_ids']),points=int(batch['feats'].shape[0])))
            yield batch

class Writer:
    def add_scalar(self,*args,**kwargs):pass

def before_forward(module,args):
    backbone=[m for block in (module.input_conv,module.unet,module.output_layer)
              for m in block.modules() if isinstance(m,nn.BatchNorm1d)]
    decoder=[m for m in module.dec.modules() if isinstance(m,nn.BatchNorm1d)]
    assert module.training and backbone and decoder and all(m.training for m in decoder)
    assert all(m.training==(row['id']!='AdaRES-Y58') for m in backbone)
    assert module.text_encoder.training==(row['id']!='AdaRES-Y59')
    assert all(not p.requires_grad for p in module.text_encoder.parameters())
    path_checks.append(True)

def after_forward(module,args,value):
    loss=value[0]
    assert torch.isfinite(loss).all() and loss.requires_grad
    losses.append(float(loss.detach().cpu()))

step=optimizer.step
def checked_step(*args,**kwargs):
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    indicator=[p.grad for p in model.dec.indi_embedding.parameters() if p.grad is not None]
    assert indicator and any(g.abs().max()>0 for g in indicator)
    gradients.append(dict(tensors=len(active),indicator_tensors=len(indicator),all_finite=True))
    return step(*args,**kwargs)

optimizer.step=checked_step
hooks=[model.register_forward_pre_hook(before_forward),model.register_forward_hook(after_forward)]
torch.cuda.reset_peak_memory_stats()
try:
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer())
    torch.cuda.synchronize()
    assert len(losses)==len(gradients)==len(batches)==len(path_checks)==2
    peak=int(torch.cuda.max_memory_reserved())
    assert peak<24*2**30
finally:
    for hook in hooks:hook.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
output=dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],
    goal_sha256=FROZEN['goal_sha256'],steps=len(gradients),losses=losses,
    native_data_loaded=True,backward_finite=True,intended_path_observed=all(path_checks),
    split='train',data_source=dict(config_data_root=cfg.data.train.data_root,
        dataset_class=type(dataset).__module__+'.'+type(dataset).__name__),
    batches=batches,gradient_checks=gradients,peak_reserved_bytes=peak,
    native_entry='tools/train.py::train',native_YAML_sha256=row['config_data']['sha256'],
    optimizer_steps=2,model_weights_saved=False,validation_used=False,admission=False,
    limitation='Two-step initialization smoke only, not trained efficacy or a full-dataset resource guarantee.')
Path(P['output']).write_text(json.dumps(output),encoding='utf-8')
