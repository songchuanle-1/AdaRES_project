"""Construct independent YAML-resolved models and load real TRAIN data on CPU."""

import base64
import copy
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import gorilla
import torch
from torch import nn
import yaml

assert P['physical_gpu'] is None
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)
root=Path(P['root'])
stage=Path(P['stage'])/'cpu_packages'
assert not stage.exists()
stage.mkdir()
native=importlib.import_module('train')
assert hashlib.sha256(Path(native.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==PAYLOAD['train_source_lf_sha256']
parent_cfg=yaml.safe_load((root/PAYLOAD['parent_config']).read_text())
rows=[]
for row in PAYLOAD['variants']:
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
    assert yaml.safe_load(config_path.read_text())==parent_cfg
    cfg=gorilla.Config.fromfile(str(config_path))
    assert cfg.train.seed==1999 and cfg.train.epochs==30
    assert cfg.dataloader.train.batch_size==2 and cfg.model.sampling_module.num_proposal==128
    assert cfg.data.train.training and cfg.data.train.prefix=='train'
    namespace='ipdn.model_paper5.'+row['package']
    importlib.import_module('ipdn.model_paper5')
    assert namespace not in sys.modules
    spec=importlib.util.spec_from_file_location(namespace,str(package/'__init__.py'),
        submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec)
    sys.modules[namespace]=module
    spec.loader.exec_module(module)
    Candidate=native.resolve_model_class(str(config_path))
    assert Candidate is module.IPDN and Candidate.__module__==namespace+'.ipdn'
    gorilla.set_random_seed(cfg.train.seed)
    model=Candidate(**cfg.model)
    assert all(p.device.type=='cpu' for p in model.parameters())
    logger=importlib.import_module('ipdn.utils').get_root_logger()
    native.load_shape_compatible_pretrain(model,cfg.train.pretrain,logger)
    dataset=native.resolve_train_dataset(str(config_path),cfg.data.train.copy(),logger)
    gorilla.set_random_seed(cfg.train.seed)
    batch=dataset.collate_fn([dataset[0],dataset[1]])
    assert len(batch['scan_ids'])==2 and len(batch['ann_ids'])>=2
    assert torch.isfinite(batch['feats']).all() and batch['lang_tokenss'].device.type=='cpu'
    model.train(True)
    backbone=[m for block in (model.input_conv,model.unet,model.output_layer)
              for m in block.modules() if isinstance(m,nn.BatchNorm1d)]
    decoder=[m for m in model.dec.modules() if isinstance(m,nn.BatchNorm1d)]
    assert backbone and decoder and all(m.training for m in decoder)
    expected_backbone=row['id']!='AdaRES-Y58'
    assert all(m.training==expected_backbone for m in backbone)
    assert model.text_encoder.training==(row['id']!='AdaRES-Y59')
    assert all(not p.requires_grad for p in model.text_encoder.parameters())
    assert all(m.weight.requires_grad and m.bias.requires_grad for m in backbone)
    first_bn=backbone[0]
    mean_before=first_bn.running_mean.clone()
    updates_before=first_bn.num_batches_tracked.clone()
    first_bn(torch.full((8,first_bn.num_features),3.0))
    mean_changed=not torch.equal(mean_before,first_bn.running_mean)
    assert mean_changed==expected_backbone
    assert int(first_bn.num_batches_tracked-updates_before)==int(expected_backbone)
    frozen_repeat=None
    trainable_encoder_preserved=None
    if row['id']=='AdaRES-Y59':
        ids=batch['lang_tokenss'][:4]
        mask=batch['lang_masks'][:4]
        with torch.no_grad():
            a=model.text_encoder(ids,attention_mask=mask)[0]
            b=model.text_encoder(ids,attention_mask=mask)[0]
        assert torch.equal(a,b) and torch.isfinite(a).all()
        frozen_repeat=0.0
        first=next(model.text_encoder.parameters())
        first.requires_grad_(True)
        model.train(True)
        trainable_encoder_preserved=model.text_encoder.training
        assert trainable_encoder_preserved
        first.requires_grad_(False)
    model.train(False)
    assert all(not m.training for m in model.modules())
    model.train(True)
    assert model.training and model.dec.training
    assert all(m.training==expected_backbone for m in backbone)
    assert model.text_encoder.training==(row['id']!='AdaRES-Y59')
    rows.append(dict(id=row['id'],config=row['config'],class_module=Candidate.__module__,
        native_YAML_semantically_unchanged=True,full_model_CPU_constructed=True,
        released_backbone_loaded=True,configured_TRAIN_data_collated=True,
        dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,
        dataset_source=str(Path(inspect.getfile(type(dataset))).resolve()),
        dataset_sha256=hashlib.sha256(Path(inspect.getfile(type(dataset))).read_bytes()).hexdigest(),
        points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),
        backbone_BN_modules=len(backbone),backbone_BN_training=expected_backbone,
        decoder_BN_modules=len(decoder),decoder_BN_preserved_training=True,
        backbone_BN_affine_trainable=True,backbone_running_statistics_updated=mean_changed,
        frozen_encoder_training=model.text_encoder.training,
        frozen_encoder_repeat_error=frozen_repeat,trainable_encoder_dropout_preserved=trainable_encoder_preserved,
        train_eval_train_roundtrip=True,parameters=sum(p.numel() for p in model.parameters()),
        native_sparse_forward=False,native_task_backward=False,admission=False))
    del model,dataset,batch,backbone,decoder,first_bn
    if row['id']=='AdaRES-Y59':del a,b,first
    gc.collect()
assert len(rows)==2 and rows[0]['parameters']==rows[1]['parameters']
assert not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,rows=rows,
    implementation_test_passed=True,actual_native_YAML_resolver_used=True,
    full_model_CPU_import_and_pretrain=True,CUDA_initialized=False,
    native_smoke_passed=False,admission=False,
    limitation='CPU construction, real configured TRAIN collate and mode-path checks only; '
               'two-step native sparse loss/backward, GPU peak and actual optimizer smoke remain pending.')),
    encoding='utf-8')
