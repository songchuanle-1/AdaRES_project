from probe_adares_y15_resume import rng, restore

def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return type(value)((clone(v) for v in value))
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    return value

def before_decoder(module, inputs):
    capture['dec'] = clone(inputs)
    capture['dec_rng'] = rng()
'Native Y45 final decomposition activity, controls and disabled W2 identity.'
import gc
import hashlib
import json
from pathlib import Path
import types
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class
assert P['physical_gpu'] == 1
torch.set_num_threads(4)
config = 'configs_song/model_paper5/adares_y45_metric_decomposition_res.yaml'
cfg = gorilla.Config.fromfile(config)
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
indices = [0, 44, 60, 101, 115, 126, 137, 171]
assert len({dataset.scene_inputs[i][0]['scene_id'] for i in indices}) == 8
sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('ipdn/model_paper5/adares_y45_metric_decomposition').glob('*.py')}
mode, capture = ('enabled', {})
model = resolve_model_class(config)(**cfg.model).cuda()
checkpoint = torch.load('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', map_location='cpu')
loaded = model.load_state_dict(checkpoint['model'], strict=False)
new = {n: p for n, p in model.named_parameters() if '.metric_readout.' in n}
assert len(new) == 4 and set(loaded.missing_keys) == set(new)
assert not set(loaded.unexpected_keys) - {'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'}
del checkpoint
model.train()

def before_criterion(module, inputs):
    pred = inputs[0]
    if mode == 'wrong_truth':
        pred['point_positive'] = pred['point_positive'].roll(1, 0)
    capture['pred'] = pred
    capture['criterion_args'] = clone(inputs[1:])

def before_readout(module, inputs):
    if mode == 'wrong_masks':
        q, masks, padding, mass, features = inputs
        return (q, masks.roll(1, 1), padding, mass, features)

def decomposition_targets(pred):
    binary = pred['masks'].detach().gt(0).float()
    valid = (~pred['batch_mask']).float()
    positive = pred['point_positive'] * valid
    negative = (pred['point_mass'] - positive) * valid
    tp = torch.einsum('bqn,bn->bq', binary, positive)
    fp = torch.einsum('bqn,bn->bq', binary, negative)
    fn = positive.sum(-1, keepdim=True) - tp
    target = torch.stack((tp, fp, fn), -1)
    return target / target.sum(-1, keepdim=True).clamp_min(1)
original_decomposition = model.criterion.loss_decomposition

def likelihood(self, pred):
    if mode == 'collapsed':
        target = decomposition_targets(pred).detach()[..., 0]
        logits = pred['error_logits']
        binary = torch.stack((torch.logsumexp(logits[..., 1:], -1), logits[..., 0]), -1)
        return -(torch.stack((1 - target, target), -1) * binary.log_softmax(-1)).sum(-1).mean()
    return original_decomposition(pred)
model.criterion.loss_decomposition = types.MethodType(likelihood, model.criterion)
hooks = [model.criterion.register_forward_pre_hook(before_criterion), model.dec.metric_readout.register_forward_pre_hook(before_readout), model.dec.register_forward_pre_hook(before_decoder)]
rows = []
for index in indices:
    gorilla.set_random_seed(1999)
    batch = dataset.collate_fn([dataset[index]])
    model.zero_grad(set_to_none=True)
    mode = 'enabled'
    gorilla.set_random_seed(1999)
    loss, losses = model(batch, mode='loss')
    loss.backward()
    norms = {n: float(p.grad.norm()) if p.grad is not None else 0 for n, p in new.items()}
    assert all((v > 0 for v in norms.values()))
    assert torch.isfinite(loss) and all((torch.isfinite(p.grad).all() for p in new.values()))
    pred = capture['pred']
    target = decomposition_targets(pred).detach()
    full_logits = pred['error_logits'].detach().clone()
    indi = pred['indis'].detach().clone()
    expected = []
    labels = batch['superpoints'].cpu().long()
    for b, truth in enumerate(batch['gt_pmasks']):
        truth = truth.cpu().reshape(-1).gt(0.5)
        masks = pred['masks'][b, :, ~pred['batch_mask'][b]].detach().cpu().gt(0)
        literal = masks[:, labels]
        triple = torch.stack(((literal & truth).sum(-1), (literal & ~truth).sum(-1), (~literal & truth).sum(-1)), -1).double()
        expected.append(triple / triple.sum(-1, keepdim=True).clamp_min(1))
    reference = torch.stack(expected).float().cuda()
    assert torch.equal(target, reference)
    original_total = float(loss)
    original_new = float(losses['decomposition_loss'])
    point_positive = pred['point_positive'].detach().clone()
    replay_args = clone(capture['criterion_args'])
    decoder_inputs = clone(capture['dec'])
    decoder_rng = capture['dec_rng']
    del loss, losses, pred, target
    model.zero_grad(set_to_none=True)
    controls = {}
    for mode in ('repeat', 'wrong_truth', 'collapsed', 'wrong_masks', 'disabled'):
        gorilla.set_random_seed(1999)
        model.dec.metric_readout.enabled = mode != 'disabled'
        with torch.no_grad():
            restore(decoder_rng)
            cached_pred = model.dec(*decoder_inputs)
            cached_pred['point_positive'] = point_positive
            loss, losses = model.criterion(cached_pred, *replay_args)
        pred = capture['pred']
        controls[mode] = dict(native_total_loss=float(loss), new_likelihood=float(losses.get('decomposition_loss', 0)), indicator_change=float((indi - pred['indis']).abs().max()))
        if mode == 'repeat':
            assert controls[mode]['indicator_change'] == 0
            assert float(loss) == original_total
            assert torch.equal(full_logits, pred['error_logits'])
        if mode == 'wrong_truth':
            assert controls[mode]['indicator_change'] == 0
        del pred, loss, losses
    model.dec.metric_readout.enabled = True
    mode = 'enabled'
    rows.append(dict(dataset_index=index, scan_id=batch['scan_ids'][0], native_total_loss=original_total, new_likelihood=original_new, literal_target_error=0, all_new_gradients=norms, controls=controls))
    del full_logits, indi, reference, batch
    capture.clear()
    gc.collect()
    torch.cuda.empty_cache()
for hook in hooks:
    hook.remove()
model.criterion.loss_decomposition = original_decomposition
model.dec.metric_readout.enabled = False
checkpoint = torch.load('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', map_location='cpu')
loaded = model.load_state_dict(checkpoint['model'], strict=False)
assert set(loaded.missing_keys) == set(new)
del checkpoint
model.eval()
dc = cfg.data.train.copy()
dc.training = dc.aug = dc.with_elastic = False
dataset = build_dataset(dc, get_root_logger())
batches = [dataset.collate_fn([dataset[i]]) for i in indices[:4]]
with torch.no_grad():
    outputs = []

    def collect_inputs(module, inputs):
        capture['physical_args'] = clone(inputs)
    hook = model.dec.register_forward_pre_hook(collect_inputs)
    for batch in batches:
        gorilla.set_random_seed(1999)
        result = model(batch, mode='predict')
        inputs = clone(capture['physical_args'])
        disabled = model.dec(*inputs)
        outputs.append((inputs, {k: disabled[k].detach().cpu().clone() for k in ('masks', 'indis', 'sample_inds')}))
    hook.remove()
del model
gc.collect()
torch.cuda.empty_cache()
parent_config = 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
parent = resolve_model_class(parent_config)(**cfg.model).cuda()
checkpoint = torch.load('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', map_location='cpu')
loaded = parent.load_state_dict(checkpoint['model'], strict=False)
assert not loaded.missing_keys
del checkpoint
parent.eval()
with torch.no_grad():
    for inputs, expected in outputs:
        result = parent.dec(*inputs)
        for key, value in expected.items():
            assert torch.equal(value, result[key].cpu()), key
assert all((hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in sources.items()))
result = dict(complete=True, id='AdaRES-Y45', rows=rows, native_scenes=8, new_tensors=4, native_queries=128, disabled_physical_W2_equal=True, literal_target_max_error=0, minimum_native_new_gradient=min((v for r in rows for v in r['all_new_gradients'].values())), scientific_status='native_controls_and_activity_not_admission', limitations=['Native W2-trained shared weights plus initialized new readout; no released-init checks yet.', 'Correct/wrong/collapsed losses may all learn; no necessity or task-benefit inference.', 'No optimizer updates, restoration, resource gate, commit or admission here.', 'No native checkpoint modified; disabled physical parent compared on frozen four TRAIN scenes.'], control_scope='Actual native cached decoder inputs and pre-decoder RNG; full-input repeat failure retained. No whole-pipeline bitwise claim.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
