def coordinate_pre(module, args):
    cache['coordinates'] = module.get_batches(args[5], args[2])[0].detach()
'Prediction-blind native TRAIN64 final-token incidence and interventions.'
import ast
import hashlib
from pathlib import Path
import statistics
import gorilla
import torch
from torch import nn
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class
torch.set_num_threads(4)
marker = Path('.codex_source_commit').read_text().strip()
assert marker == 'b49b744dc2943a38ab4c3aeb788acc6a2cb90011'
config = 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
cfg = gorilla.Config.fromfile(config)
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
model = resolve_model_class(config)(**cfg.model).cuda()
checkpoint = '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth'
assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
loaded = model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=False)
assert not loaded.missing_keys and (not set(loaded.unexpected_keys) - {'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'})
source = Path('ipdn/model_paper5/adares_y36_local_adjacency/dec.py').read_bytes()
tree = ast.parse(source.decode())
node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MaskLocalAdjacencyReadout'))
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[node], type_ignores=[]), 'y33_actual_class', 'exec'), namespace)
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(1999)
    readout = namespace['MaskLocalAdjacencyReadout'](256).cuda()
indices, seen = ([], set())
for index, chunk in enumerate(dataset.scene_inputs):
    if chunk[0]['scene_id'] not in seen:
        seen.add(chunk[0]['scene_id'])
        indices.append(index)
    if len(indices) == 64:
        break
assert len(indices) == 64
model.train()
cache = {}

def scene_post(module, args, output):
    cache['scene'] = output.detach()

def indicator_pre(module, args):
    cache['indicator'] = tuple((v.detach() for v in args))
hooks = [model.dec.ila_ffn_layers[-1].register_forward_hook(scene_post), model.dec.indi_embedding.register_forward_pre_hook(indicator_pre), model.dec.register_forward_pre_hook(coordinate_pre)]
rows = []
with torch.no_grad():
    for index in indices:
        cache.clear()
        batch = dataset.collate_fn([dataset[index]])
        model(batch, mode='loss')
        query, masks, padding, mass = cache['indicator']
        tokens = cache['scene']
        coords = cache['coordinates']
        valid = (~padding).to(masks.dtype)
        weights = masks.sigmoid() * valid[:, None] * mass[:, None]
        means = torch.bmm(weights, tokens) / weights.sum(-1, keepdim=True).clamp_min(1e-06)
        variance = (torch.bmm(weights, tokens.square()) / weights.sum(-1, keepdim=True).clamp_min(1e-06) - means.square()).clamp_min(0)
        direct = readout(query, masks, padding, mass, tokens, coords)
        repeat = readout(query, masks, padding, mass, tokens, coords)
        shuffled_tokens = tokens.clone()
        for b in range(tokens.shape[0]):
            count = int(valid[b].sum())
            shuffled_tokens[b, :count] = tokens[b, :count].roll(1, 0)
        wrong_tokens = readout(query, masks, padding, mass, shuffled_tokens, coords)
        wrong_masks = readout(query, masks.roll(1, 1), padding, mass, tokens, coords)
        perm = torch.arange(tokens.shape[1] - 1, -1, -1, device=tokens.device)
        joint = readout(query, masks[:, :, perm], padding[:, perm], mass[:, perm], tokens[:, perm], coords[:, perm])
        original = model.dec.indi_embedding(query, masks, padding, mass)
        binary = masks.sigmoid().ge(0.5)
        sp = batch['superpoints'].long().cuda()
        counts = torch.bincount(sp)
        positive = torch.stack([torch.bincount(sp[gt.cuda().bool()], minlength=counts.numel()) for gt in batch['gt_pmasks']])
        intersection = (binary * positive[:, None]).sum(-1)
        union = ((binary * counts).sum(-1) + positive.sum(-1)[:, None] - intersection).clamp_min(1)
        quality = intersection / union
        choice = original.softmax(-1)[..., 1].argmax(-1)
        selected = quality.gather(1, choice[:, None]).squeeze(1)
        gap = quality.max(-1).values - selected
        row = dict(scan_id=batch['scan_ids'][0], dataset_index=index, expressions=len(batch['ann_ids']), oracle_gap=gap.cpu().tolist(), token_variance_mean=float(variance.mean()), attention_wrong_tokens_max=float((direct - wrong_tokens).abs().max()), attention_wrong_masks_max=float((direct - wrong_masks).abs().max()), repeat_max=float((direct - repeat).abs().max()), support_permutation_max=float((direct - joint).abs().max()), finite=bool(torch.isfinite(direct).all()))
        assert row['finite'] and row['repeat_max'] == 0
        assert row['support_permutation_max'] < 1e-05
        rows.append(row)
        if len(rows) % 8 == 0:
            print(json.dumps(dict(scans=len(rows))), flush=True)
        cache.clear()
        del batch
for hook in hooks:
    hook.remove()
result = dict(complete=True, id='AdaRES-Y36', rows=rows, scans=64, expressions=sum((r['expressions'] for r in rows)), expressions_with_oracle_gap=sum((g > 0 for r in rows for g in r['oracle_gap'])), scene_mean_oracle_gap=statistics.mean((statistics.mean(r['oracle_gap']) for r in rows)), positive_variance_scans=sum((r['token_variance_mean'] > 0 for r in rows)), wrong_tokens_nonzero_scans=sum((r['attention_wrong_tokens_max'] > 0 for r in rows)), wrong_masks_nonzero_scans=sum((r['attention_wrong_masks_max'] > 0 for r in rows)), minimum_wrong_token_delta=min((r['attention_wrong_tokens_max'] for r in rows)), repeat_max=max((r['repeat_max'] for r in rows)), permutation_max=max((r['support_permutation_max'] for r in rows)), source_sha256=hashlib.sha256(source).hexdigest(), source_marker=marker, checkpoint=checkpoint, gpu_precheck=dict(memory_mib=used, utilization=util), limitations=['TRAIN incidence and random initialized readout, not efficacy or necessity.', 'GT point-IoU used only for diagnosis, never inference input.', 'Upstream attention exists; no global blindness or novelty claim.', 'No validation/numeric search, old source/checkpoint edits or GPU0 query.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
