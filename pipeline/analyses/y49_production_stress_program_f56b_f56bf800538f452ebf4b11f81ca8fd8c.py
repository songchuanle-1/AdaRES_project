"""Frozen native batch2 stress for Y46 revision2, with paired fresh optimizers."""
import copy, gc, hashlib, json, time
from pathlib import Path
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class
assert P['physical_gpu'] == 1 and torch.cuda.device_count() == 1
torch.set_num_threads(4)
root = Path(P['root'])
decoder = root / 'ipdn/model_paper5/adares_y49_resolvent_coherence/dec.py'
assert hashlib.sha256(decoder.read_bytes()).hexdigest() == '50cfd1e821c188ddcb44bb929a6ffb84a186f0af89e1deb002aae10ca7e3bf31'
config = 'configs_song/model_paper5/adares_y49_resolvent_coherence_res.yaml'
cfg = gorilla.Config.fromfile(config)
assert cfg.train.epochs == 30 and cfg.train.seed == 1999
gorilla.set_random_seed(1999)
model = resolve_model_class(config)(**cfg.model).cuda()
checkpoint = '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth'
assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
saved = torch.load(checkpoint, map_location='cpu')
loaded = model.load_state_dict(saved['model'], strict=False)
assert set(loaded.missing_keys) == {'dec.mask_resolvent_readout.' + n for n, _ in model.dec.mask_resolvent_readout.named_parameters()}
assert not set(loaded.unexpected_keys) - {'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'}
del saved
starting = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
model.train()
result = dict(complete=False, id='AdaRES-Y49', implementation_revision=1, rows=[], added_parameters=66561, limitations=['W2 inherited state plus unchanged new Y46 readout, not trained Y46 efficacy.', 'Same production model with readout disabled is paired parent; unused extra head remains allocated.', 'Each arm uses identical initial state and a fresh native AdamW optimizer,2warmups/3readings.', 'Transient optimizer updates remain memory-only, no original checkpoint/source modification.', 'Time/memory are observed, not automatically a lightweight Pareto PASS.'])
for indices in ([0, 44], [728, 6780]):
    gorilla.set_random_seed(1999)
    batch = dataset.collate_fn([dataset[i] for i in indices])
    row = dict(indices=indices, scan_ids=batch['scan_ids'], points=len(batch['superpoints']))
    for name, enabled in (('parent', False), ('candidate', True)):
        model.load_state_dict(starting)
        model.dec._resolvent_attachment.enabled = enabled
        optimizer = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))

        def step():
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(batch, mode='loss')
            assert torch.isfinite(loss)
            loss.backward()
            assert all((torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None))
            if enabled:
                assert all((v.grad is not None and float(v.grad.norm()) > 0 for v in model.dec.mask_resolvent_readout.parameters()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            return float(loss)
        for _ in range(2):
            step()
        readings = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            loss = step()
            torch.cuda.synchronize()
            readings.append(dict(seconds=time.monotonic() - start, loss=loss, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30))
        row[name] = readings
        del optimizer
        model.zero_grad(set_to_none=True)
        model.dec._resolvent_attachment.cached = []
        model.dec._resolvent_attachment.cached = []
        gc.collect()
        torch.cuda.empty_cache()
    result['rows'].append(row)
    Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
    del batch
assert hashlib.sha256(decoder.read_bytes()).hexdigest() == '50cfd1e821c188ddcb44bb929a6ffb84a186f0af89e1deb002aae10ca7e3bf31'
result['complete'] = True
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
