assert __import__('pathlib').Path('.codex_source_commit').read_text().strip()=='67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773'
"""Matched serial native Y39/W2 training and inference resource probes."""
import copy
import gc
from pathlib import Path
import statistics
import time
import gorilla
import numpy as np
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain
torch.set_num_threads(4)
config = 'configs_song/model_paper5/adares_y39_point_distribution_res.yaml'
parent = 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
cfg = gorilla.Config.fromfile(config)
logger = get_root_logger()
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), logger)
indices, seen = ([], set())
for index, chunk in enumerate(dataset.scene_inputs):
    if chunk[0]['scene_id'] not in seen:
        seen.add(chunk[0]['scene_id'])
        indices.append(index)
    if len(indices) == 8:
        break
train_batches = [dataset.collate_fn([dataset[i] for i in indices[:2]]), dataset.collate_fn([dataset[728], dataset[6780]])]
rows, initial = ({}, {})
for name, model_config in (('parent', parent), ('candidate', config)):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(model_config)(**cfg.model).cuda()
    load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    opt = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
    count = sum((p.numel() for p in model.parameters()))
    if not initial:
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        assert all((torch.equal(v.cpu(), initial[k]) for k, v in model.state_dict().items() if k in initial))
        assert set(model.state_dict()) - set(initial) == {k for k in model.state_dict() if '.point_readout.' in k}
        initial.clear()

    def step(batch):
        opt.zero_grad(set_to_none=True)
        loss, _ = model(batch, mode='loss')
        assert torch.isfinite(loss)
        loss.backward()
        assert all((torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        return float(loss)
    readings = []
    for batch in train_batches:
        for _ in range(2):
            step(batch)
        samples = []
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            loss = step(batch)
            torch.cuda.synchronize()
            samples.append(dict(seconds=time.monotonic() - start, loss=loss, allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30))
        readings.append(dict(scan_ids=batch['scan_ids'], points=len(batch['superpoints']), superpoints=int(batch['batch_offsets'][-1]), samples=samples))
    rows[name] = dict(parameters=count, training=readings)
    del opt, model
    gc.collect()
    torch.cuda.empty_cache()
del initial, train_batches, dataset
dc = cfg.data.train.copy()
dc.training = False
dc.aug = False
dc.with_elastic = False
dataset = build_dataset(dc, logger)
batches = [dataset.collate_fn([dataset[i]]) for i in indices + [728, 6780]]
for name, model_config in (('parent', parent), ('candidate', config)):
    model = resolve_model_class(model_config)(**cfg.model).cuda()
    ck = torch.load('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', map_location='cpu')
    loaded = model.load_state_dict(ck['model'], strict=False)
    expected_missing = {k for k in model.state_dict() if '.point_readout.' in k}
    assert set(loaded.missing_keys) == expected_missing and (not set(loaded.unexpected_keys) - {'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'})
    del ck
    model.eval()
    readings = []
    with torch.no_grad():
        for batch in batches:
            for _ in range(2):
                gorilla.set_random_seed(1999)
                model(batch, mode='predict')
            samples = []
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                gorilla.set_random_seed(1999)
                start = time.monotonic()
                pred = model(batch, mode='predict')
                torch.cuda.synchronize()
                samples.append(dict(seconds=time.monotonic() - start, allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30))
                assert all((v.shape[0] == 128 for v in pred['indi']))
                assert all((v.numel() == len(batch['superpoints']) for v in pred['pred_pmask']))
            readings.append(dict(scan_id=batch['scan_ids'][0], points=len(batch['superpoints']), dataset_index=(indices + [728, 6780])[len(readings)], samples=samples))
    rows[name]['inference'] = readings
    del model, pred
    gc.collect()
    torch.cuda.empty_cache()
summary = {}
for name, value in rows.items():
    summary[name] = dict(parameters=value['parameters'], stress_train_median=statistics.median((s['seconds'] for s in value['training'][-1]['samples'])), stress_train_reserved=max((s['reserved_gib'] for s in value['training'][-1]['samples'])), inference_median=statistics.median((s['seconds'] for r in value['inference'] for s in r['samples'])), inference_p95=float(np.percentile([s['seconds'] for r in value['inference'] for s in r['samples']], 95)), inference_reserved=max((s['reserved_gib'] for r in value['inference'] for s in r['samples'])))
result = dict(complete=True, id='AdaRES-Y39', rows=rows, summary=summary, parameters_added=rows['candidate']['parameters'] - rows['parent']['parameters'], inference_queries=128, inference_selected=1, gpu_precheck=dict(memory_mib=used, utilization=util), limitations=['Actual native paired serial training2warmups/3readings, not trained Pareto efficacy.', 'Eight prediction-blind TRAIN scenes plus frozen maximum pair728/6780, serialbatch1 inference.', 'Three reads per scene descriptive p95, not reliable tail/significance.', 'Identical shared initial tensors verified; inference W2+new initialized trajectory readout, not trained Y39 weights.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
