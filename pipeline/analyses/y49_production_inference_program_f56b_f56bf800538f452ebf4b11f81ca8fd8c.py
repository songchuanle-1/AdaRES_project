"""Frozen production TRAIN8 controls and serial costs; no efficacy claim."""
import gc, hashlib, json, time
from pathlib import Path
import gorilla
import numpy as np
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class
assert P['physical_gpu'] == 1 and torch.cuda.device_count() == 1
torch.set_num_threads(4)
root = Path(P['root'])
config = 'configs_song/model_paper5/adares_y49_resolvent_coherence_res.yaml'
decoder = root / 'ipdn/model_paper5/adares_y49_resolvent_coherence/dec.py'
assert hashlib.sha256(decoder.read_bytes()).hexdigest() == '50cfd1e821c188ddcb44bb929a6ffb84a186f0af89e1deb002aae10ca7e3bf31'
protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for n in ('adares_w2_pointmass_competition', 'adares_y45_metric_decomposition', 'adares_y48_mask_equivalence', 'adares_y49_resolvent_coherence') for p in (root / 'ipdn/model_paper5' / n).glob('*.py')}
cfg = gorilla.Config.fromfile(config)
gorilla.set_random_seed(1999)
model = resolve_model_class(config)(**cfg.model).cuda()
checkpoint_path = '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth'
assert hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest() == 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
loaded = model.load_state_dict(checkpoint['model'], strict=False)
head = model.dec.mask_resolvent_readout
assert set(loaded.missing_keys) == {'dec.mask_resolvent_readout.' + n for n, _ in head.named_parameters()}
assert not set(loaded.unexpected_keys) - {'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'}
del checkpoint
attachment = model.dec._resolvent_attachment
buffers = {n: v.detach().cpu().clone() for n, v in model.named_buffers()}
indices = [0, 44, 60, 101, 115, 126, 137, 171]
result = dict(complete=False, id='AdaRES-Y49', implementation_revision=1, manifest_indices=indices, native_rows=[], inference_rows=[], added_parameters=sum((v.numel() for v in head.parameters())), limitations=['W2 inherited checkpoint plus unchanged new production readout, not trained Y46.', 'Cached controls are matched; no full-loader bitwise repeat claim.', 'Costs are descriptive, not resource admission. Separate native stress remains pending.', 'No source/config/checkpoint/queue modification or diagnostic weight deployment.'])

def persist():
    Path(P['output']).write_text(json.dumps(result), encoding='utf-8')

def measured(batch, enabled, mode):
    model.zero_grad(set_to_none=True)
    for n, v in model.named_buffers():
        v.copy_(buffers[n].to(v.device))
    gorilla.set_random_seed(1999)
    attachment.enabled = enabled
    attachment.wrong_correspondence = False
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    if mode == 'loss':
        loss, _ = model(batch, mode=mode)
        assert torch.isfinite(loss)
        loss.backward()
        assert all((torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None))
        value = float(loss)
    else:
        with torch.no_grad():
            prediction = model(batch, mode=mode)
        assert all((v.shape[0] == 128 for v in prediction['indi']))
        value = None
    torch.cuda.synchronize()
    return dict(seconds=time.monotonic() - start, loss=value, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)
data_cfg = cfg.data.train.copy()
data_cfg.training = False
data_cfg.aug = False
data_cfg.with_elastic = False
gorilla.set_random_seed(1999)
dataset = build_dataset(data_cfg, get_root_logger())
model.eval()
for i in indices + [728, 6780]:
    batch = dataset.collate_fn([dataset[i]])
    row = dict(dataset_index=i, scan_ids=batch['scan_ids'], points=len(batch['superpoints']))
    for name, enabled in (('parent', False), ('candidate', True)):
        for _ in range(2):
            measured(batch, enabled, 'predict')
        row[name] = [measured(batch, enabled, 'predict') for _ in range(3)]
    result['inference_rows'].append(row)
    persist()
    del batch
assert all((hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in protected.items()))
result['protected_sources_unchanged'] = True
result['complete'] = True
result['cost_only'] = True
persist()
