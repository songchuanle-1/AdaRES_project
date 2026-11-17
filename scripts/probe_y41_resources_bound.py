import hashlib
from pathlib import Path
manifest=[{'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/__init__.py', 'sha256': '8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722'}, {'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/backbone.py', 'sha256': '1eb4ac3483961e708c23b0abad396c181dbf38c56fcca559d88b9622af46bafd'}, {'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/dec.py', 'sha256': '33d27cdfd93c39f2b9889d6bf7f3abad3acd4c403dec5a14c8da0f16f6852bd9'}, {'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/ipdn.py', 'sha256': '63c8f30bda20300d98d3d9a8a6478fc1a841cbaa3b334ee71688711e06375d97'}, {'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/loss.py', 'sha256': '4aa2818948ebfd9fe416b3b310a053384508ef92c3b67831833fece560523084'}, {'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/sample_model.py', 'sha256': '03fd683acc6d49d9b53e634b42bd150a3259a1a15c7dc8fe2badf115870a345b'}, {'path': 'AdaRES/configs_song/model_paper5/adares_y41_boundary_probability_res.yaml', 'sha256': '1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03'}]
root=Path('/bak_home/songcl/Project/AdaRES_project')
assert all(hashlib.sha256((root / row['path']).read_bytes()).hexdigest()==row['sha256'] for row in manifest)
preserved_cmd=Path('/proc/102267/cmdline').read_bytes().split(b'\0')
assert b'configs_song/model_paper5/adares_y39_point_distribution_res.yaml' in preserved_cmd
assert preserved_cmd[preserved_cmd.index(b'--gpu_ids')+1]==b'1'
assert P['physical_gpu']==0
"""Matched serial native Y41/W2 training and inference resource probes."""
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
config = 'configs_song/model_paper5/adares_y41_boundary_probability_res.yaml'
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
        assert set(model.state_dict()) - set(initial) == {k for k in model.state_dict() if '.boundary_readout.' in k}
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
    expected_missing = {k for k in model.state_dict() if '.boundary_readout.' in k}
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
result = dict(complete=True, id='AdaRES-Y41', rows=rows, summary=summary, parameters_added=rows['candidate']['parameters'] - rows['parent']['parameters'], inference_queries=128, inference_selected=1, gpu_precheck=dict(memory_mib=used, utilization=util), limitations=['Actual native paired serial training2warmups/3readings, not trained Pareto efficacy.', 'Eight prediction-blind TRAIN scenes plus frozen maximum pair728/6780, serialbatch1 inference.', 'Three reads per scene descriptive p95, not reliable tail/significance.', 'Identical shared initial tensors verified; inference W2+new initialized trajectory readout, not trained Y41 weights.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')

assert Path('/proc/102267/cmdline').read_bytes().split(b'\0')==preserved_cmd
assert all(hashlib.sha256((root / row['path']).read_bytes()).hexdigest()==row['sha256'] for row in manifest)
