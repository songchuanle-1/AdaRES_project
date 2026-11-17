"""Frozen serial inference and separate native stress with trained weights."""
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
assert (root / '.codex_source_commit').read_text().strip() == '6725f9bd6a8f9b47addae6a4b8336d004f73f1ae'
assert not Path('/proc/108331').exists()
assert not subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).decode().strip(), 'GPU0 has a compute process'
source_manifest = [('ipdn/model_paper5/adares_y35_language_evidence/__init__.py', '8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722'), ('ipdn/model_paper5/adares_y35_language_evidence/backbone.py', '1eb4ac3483961e708c23b0abad396c181dbf38c56fcca559d88b9622af46bafd'), ('ipdn/model_paper5/adares_y35_language_evidence/dec.py', '06beaac9a2625dde8916e14a3e5157e700e370575ea5cb2324ae60de66282b9b'), ('ipdn/model_paper5/adares_y35_language_evidence/ipdn.py', '63c8f30bda20300d98d3d9a8a6478fc1a841cbaa3b334ee71688711e06375d97'), ('ipdn/model_paper5/adares_y35_language_evidence/loss.py', '4aa2818948ebfd9fe416b3b310a053384508ef92c3b67831833fece560523084'), ('ipdn/model_paper5/adares_y35_language_evidence/sample_model.py', '03fd683acc6d49d9b53e634b42bd150a3259a1a15c7dc8fe2badf115870a345b'), ('configs_song/model_paper5/adares_y35_language_evidence_res.yaml', '1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03')]
assert all((hashlib.sha256((root / path).read_bytes()).hexdigest() == expected for path, expected in source_manifest))
captures = {name: json.loads(Path('/tmp/adares_y35_' + mode + '_' + P['token'] + '.json').read_text()) for name, mode in (('candidate', 'y35'), ('parent', 'w2'))}
assert all((c['complete'] and c['completed_passes'] == 2 for c in captures.values()))
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
assert parameters == {'candidate': 164958087, 'parent': 164629637}
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
result = dict(complete=True, id='AdaRES-Y35', manifest_indices=indices, source_files=[dict(path=path, sha256=sha) for path, sha in source_manifest], parameters=parameters, available_queries=128, selected_queries=1, gpu_precheck=dict(memory_mib=used, utilization=util), checkpoints={name: dict(path=capture['checkpoint'], sha256=capture['checkpoint_sha256']) for name, capture in captures.items()}, rows=rows, training_stress=stress, summary=summary, limitations=['Frozen TRAIN8 plus largest-pair scenes, serial native predict batch1, two warmups/three readings.', 'Separate native loss/backward/AdamW stress batch2; trained weights, fresh native optimizer warmed twice.', 'Stress updates remain memory-only; original checkpoints are not overwritten or resumed.', 'Candidate then parent order does not isolate transient hardware effects; descriptive p95 only.', 'One trained checkpoint per method; cost is not causal efficacy, equivalence or Pareto dominance.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
