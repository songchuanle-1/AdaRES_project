import hashlib, json
from pathlib import Path
assert P['physical_gpu'] == 1
terminal = json.loads(Path('/tmp/adares_y43_terminal_7a40_' + P['token'] + '.json').read_text())
protected = terminal['source_hashes']

def check_originals():
    assert all((hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in protected.items()))
    for row in terminal['checkpoints']:
        digest = hashlib.sha256()
        with open(row['path'], 'rb') as stream:
            for block in iter(lambda: stream.read(8388608), b''):
                digest.update(block)
        assert digest.hexdigest() == row['sha256']
check_originals()
'Frozen serial inference and separate native stress with trained weights.'
import copy
import gc
import hashlib
from pathlib import Path
import statistics
import time
import gorilla
import numpy as np
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class
root = Path(P['root'])
combined = json.loads(Path('/tmp/adares_y43_frozen_392a_' + P['token'] + '.json').read_text())
captures = {'candidate': combined['rows']['y43'], 'parent': combined['rows']['w2']}
assert all((c['complete'] and c['completed_passes'] == 2 for c in captures.values()))
source_manifest = [(p, h) for p, h in json.loads(Path('/tmp/adares_y43_terminal_7a40_' + P['token'] + '.json').read_text())['source_hashes'].items()]
assert all((hashlib.sha256((root / p).read_bytes()).hexdigest() == h for p, h in source_manifest))
cfg = gorilla.Config.fromfile(captures['candidate']['config'])
torch.set_num_threads(4)

def load_trained(name):
    capture = captures[name]
    digest = hashlib.sha256()
    with open(capture['checkpoint'], 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == capture['checkpoint_sha256']
    gorilla.set_random_seed(1999)
    model = resolve_model_class(capture['config'])(**cfg.model).cuda()
    checkpoint = torch.load(capture['checkpoint'], map_location='cpu')
    loaded = model.load_state_dict(checkpoint['model'], strict=False)
    assert not loaded.missing_keys and (not set(loaded.unexpected_keys) - {'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'})
    del checkpoint
    return model
data_cfg = cfg.data.train.copy()
data_cfg.training = False
data_cfg.aug = False
data_cfg.with_elastic = False
gorilla.set_random_seed(1999)
dataset = build_dataset(data_cfg, get_root_logger())
indices, seen = ([], set())
for index, chunk in enumerate(dataset.scene_inputs):
    if chunk[0]['scene_id'] not in seen:
        seen.add(chunk[0]['scene_id'])
        indices.append(index)
    if len(indices) == 8:
        break
indices += [728, 6780]
batches = [dataset.collate_fn([dataset[index]]) for index in indices]
assert [batch['scan_ids'] for batch in batches[-2:]] == [['scene0054_00'], ['scene0497_00']]
rows, parameters = ({}, {})
for name in ('candidate', 'parent'):
    model = load_trained(name)
    model.eval()
    parameters[name] = sum((parameter.numel() for parameter in model.parameters()))
    readings = []
    with torch.no_grad():
        for index, batch in zip(indices, batches):
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
                prediction = model(batch, mode='predict')
                torch.cuda.synchronize()
                samples.append(dict(seconds=time.monotonic() - start, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30))
                assert all((value.shape[0] == 128 for value in prediction['indi']))
                assert all((value.numel() == len(batch['superpoints']) for value in prediction['pred_pmask']))
            readings.append(dict(scan_ids=batch['scan_ids'], dataset_index=index, points=len(batch['superpoints']), readings=samples))
    rows[name] = readings
    del model, prediction
    gc.collect()
    torch.cuda.empty_cache()
assert parameters == {name: capture['parameters'] for name, capture in captures.items()}
del batches, dataset
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
stress_batches = [dataset.collate_fn([dataset[index] for index in indices[:2]]), dataset.collate_fn([dataset[728], dataset[6780]])]
stress = {}
for name in ('candidate', 'parent'):
    model = load_trained(name)
    model.train()
    optimizer = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))

    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(batch, mode='loss')
        assert torch.isfinite(loss)
        loss.backward()
        assert all((torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        return float(loss)
    readings = []
    for batch in stress_batches:
        for _ in range(2):
            step(batch)
        samples = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            loss = step(batch)
            torch.cuda.synchronize()
            samples.append(dict(seconds=time.monotonic() - start, loss=loss, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30))
        readings.append(dict(scan_ids=batch['scan_ids'], points=len(batch['superpoints']), readings=samples))
    stress[name] = readings
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
summary = {}
for name, value in rows.items():
    summary[name] = {}
    for group, chosen in (('train8', value[:8]), ('largest_pair', value[-2:])):
        samples = [sample for row in chosen for sample in row['readings']]
        seconds = [sample['seconds'] for sample in samples]
        summary[name][group] = dict(median_seconds=statistics.median(seconds), descriptive_p95_seconds=float(np.percentile(seconds, 95)), peak_allocated_gib=max((sample['peak_allocated_gib'] for sample in samples)), peak_reserved_gib=max((sample['peak_reserved_gib'] for sample in samples)))
    samples = stress[name][-1]['readings']
    summary[name]['native_train_stress'] = dict(median_seconds=statistics.median((sample['seconds'] for sample in samples)), peak_allocated_gib=max((sample['peak_allocated_gib'] for sample in samples)), peak_reserved_gib=max((sample['peak_reserved_gib'] for sample in samples)))
result = dict(complete=True, id='AdaRES-Y43', manifest_indices=indices, source_files=[dict(path=path, sha256=sha) for path, sha in source_manifest], parameters=parameters, available_queries=128, selected_queries=1, gpu_precheck=dict(memory_mib=used, utilization=util), checkpoints={name: dict(path=capture['checkpoint'], sha256=capture['checkpoint_sha256']) for name, capture in captures.items()}, rows=rows, training_stress=stress, summary=summary, limitations=['Frozen TRAIN8 plus largest-pair scenes, serial native predict batch1, two warmups/three readings.', 'Separate native loss/backward/AdamW stress batch2; trained weights, fresh native optimizer warmed twice.', 'Stress updates remain memory-only; original checkpoints are not overwritten or resumed.', 'Candidate then parent order does not isolate transient hardware effects; descriptive p95 only.', 'One trained checkpoint per method; cost is not causal efficacy, equivalence or Pareto dominance.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
check_originals()
