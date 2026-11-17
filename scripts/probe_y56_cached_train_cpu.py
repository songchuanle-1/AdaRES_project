"""CPU-only replay of hash-bound native TRAIN indicator inputs and a trained head."""
import ast
import base64
import gc
from pathlib import Path
import zlib

import numpy as np


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unpack(item):
    raw = zlib.decompress(base64.b64decode(item['zlib_base64'], validate=True))
    assert hashlib.sha256(raw).hexdigest() == item['sha256']
    array = np.frombuffer(raw, dtype=np.dtype(item['dtype'])).reshape(item['shape']).copy()
    assert np.isfinite(array).all()
    return torch.from_numpy(array)


def incidence(query, masks):
    q, m = query.detach().cpu().numpy(), masks.detach().cpu().numpy()
    _, inverse, counts = np.unique(np.concatenate([q, m], axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    return dict(query_unique=int(len(np.unique(q, axis=0))),
        soft_mask_unique=int(len(np.unique(m, axis=0))),
        joint_unique=int(len(counts)), hard_mask_unique=int(len(np.unique(m > 0, axis=0))),
        joint_duplicate_excess=int(len(q) - len(counts)),
        hard_mask_duplicate_excess=int(len(q) - len(np.unique(m > 0, axis=0)))), inverse, counts


assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                 if line.startswith('MemAvailable:'))
assert available >= 8 * 1024 ** 3, 'Do not pressure host RAM while native runs train'
checkpoint = Path(PAYLOAD['checkpoint'])
assert digest_file(checkpoint) == PAYLOAD['checkpoint_sha256']
raw_state = torch.load(str(checkpoint), map_location='cpu')
prefix = 'dec.indi_embedding.'
head_state = {key[len(prefix):]: value.detach().clone() for key, value in raw_state['model'].items()
              if key.startswith(prefix)}
assert head_state
checkpoint_meta = {key: raw_state.get('meta', {}).get(key) for key in ('epoch', 'best_miou')}
del raw_state
gc.collect()
model_class = Indicator


def Indicator(width):
    model = model_class(width)
    model.load_state_dict(head_state, strict=True)
    return model


rows = []
exec(compile(PAYLOAD['fixture_loop'], '<frozen-Y56-fixture-loop-trained-head>', 'exec'))
fixture_rows = rows
torch.manual_seed(1999)
model = Indicator(256).eval()
initial_state = state_hash(model)
records = []
for scene in PAYLOAD['scenes']:
    for expression in scene['expressions']:
        inputs = expression['packed_native_inputs']
        query, masks, mass = (unpack(inputs[key]).float() for key in ('query', 'masks', 'masses'))
        assert query.shape == (128, 256) and masks.shape[0] == 128
        assert masks.shape[1] == len(mass) and bool((mass >= 0).all())
        stats, inverse, counts = incidence(query, masks)
        with torch.no_grad():
            logits, context = native(model, query.unsqueeze(0), masks.unsqueeze(0), mass.unsqueeze(0))
            repeated, _ = native(model, query.unsqueeze(0), masks.unsqueeze(0), mass.unsqueeze(0))
            order = torch.arange(127, -1, -1)
            permuted, _ = native(model, query[order].unsqueeze(0), masks[order].unsqueeze(0), mass.unsqueeze(0))
        bound = 128 * torch.finfo(query.dtype).eps * (1 + max(float(logits.abs().max()), float(context.abs().max())))
        repeat_error = float((logits - repeated).abs().max())
        permutation_error = float((logits - permuted[:, order]).abs().max())
        assert max(repeat_error, permutation_error) <= bound
        singleton = [i for i, group in enumerate(inverse) if counts[group] == 1]
        transfer = dict(applicable=False, reason='No eligible exact duplicate donor and distinct competitor with a singleton focal query')
        if singleton:
            focal = singleton[0]
            focal_group = inverse[focal]
            donors = [g for g, count in enumerate(counts) if count >= 2 and g != focal_group]
            if donors:
                donor = donors[0]
                receivers = [g for g in range(len(counts)) if g not in (donor, focal_group)]
                if receivers:
                    receiver = receivers[0]
                    slot = int(np.flatnonzero(inverse == donor)[0])
                    representative = int(np.flatnonzero(inverse == receiver)[0])
                    changed_q, changed_m = query.clone(), masks.clone()
                    changed_q[slot], changed_m[slot] = query[representative], masks[representative]
                    assert torch.equal(changed_q[focal], query[focal]) and torch.equal(changed_m[focal], masks[focal])
                    changed_stats, _, _ = incidence(changed_q, changed_m)
                    assert changed_stats['joint_unique'] == stats['joint_unique']
                    with torch.no_grad():
                        output, output_context = native(model, changed_q.unsqueeze(0), changed_m.unsqueeze(0), mass.unsqueeze(0))
                    transfer = dict(applicable=True, focal=focal, donor_group=int(donor), receiver_group=int(receiver),
                        focal_logit_difference=float((logits[:, focal] - output[:, focal]).abs().max()),
                        focal_context_difference=float((context[:, focal] - output_context[:, focal]).abs().max()),
                        query_count_preserved=True, unique_query_mask_set_preserved=True)
        records.append(dict(scene=scene['scan_id'], dataset_index=scene['dataset_index'], ann_id=expression['ann_id'],
            native_queries=128, valid_superpoints=len(mass), **stats, repeat_error=repeat_error,
            permutation_error=permutation_error, arithmetic_screen_bound=bound, multiplicity_transfer=transfer))
assert len(records) == 32 and len(PAYLOAD['scenes']) == 8
assert state_hash(model) == initial_state
assert digest_file(checkpoint) == PAYLOAD['checkpoint_sha256']
active = [row for row in fixture_rows if row['fixture'] == 'contrastive_competitors']
result = dict(complete=True, id='AdaRES-Y56', device='cpu', threads=torch.get_num_threads(),
    cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], checkpoint_meta=checkpoint_meta,
    checkpoint_sha256=PAYLOAD['checkpoint_sha256'], capture_sha256=PAYLOAD['capture_sha256'],
    cached_native_TRAIN_scenes=8, cached_native_TRAIN_expressions=32, capture_unit='Final indicator inputs, single-scene TRAIN loss forward; not all7layers or native batch2',
    trained_head_fixture_signal=all(row['native_focal_logit_delta'] > row['numerical_bound'] for row in active),
    exact_joint_duplicate_excess=sum(row['joint_duplicate_excess'] for row in records),
    hard_mask_duplicate_excess=sum(row['hard_mask_duplicate_excess'] for row in records),
    applicable_counterfactuals=sum(row['multiplicity_transfer']['applicable'] for row in records),
    max_repeat_error=max(row['repeat_error'] for row in records),
    max_permutation_error=max(row['permutation_error'] for row in records),
    fixture_rows=fixture_rows, records=records, model_state_unchanged=True, checkpoint_file_unchanged=True,
    gpu_tests=0, training_launches=0, admission=False, full_native_backward=False,
    limitations=[
        'Cached32TRAIN expressions from8fixed scenes, native sparse/FPS single-scene final indicator inputs only; no population or all-layer conclusion.',
        'Original capture source loads the W2 checkpoint path but did not record a checkpoint blob hash at capture time; current CPU replay binds the verified frozen W2 checkpoint SHA. No retroactive provenance claim.',
        'Exact query/soft-mask duplication and native zero-logit hard-mask equivalence are separately counted. Hard-mask equivalence with different features is not an exact duplicate of a competitor representation.',
        'No exact-duplicate counterfactual is forced when no eligible donor exists. Synthetic trained-head sensitivity is not actual incidence, trained efficacy, native resource/backward PASS or novelty.',
        'No source, config, checkpoint, running process, targets or selector changed; no occupancy/loss/threshold/epoch search.',
    ])
print(json.dumps(result))
