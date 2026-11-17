"""Actual prospective Y36 class with fixed adjacency discrimination."""
import ast
import hashlib
from pathlib import Path
import torch
from torch import nn
torch.set_num_threads(4)
assert not torch.cuda.is_available()
source = Path('ipdn/model_paper5/adares_y36_local_adjacency/dec.py').read_bytes()
tree = ast.parse(source.decode('utf-8'))
node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MaskLocalAdjacencyReadout'))
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[node], type_ignores=[]), 'y36_actual_class', 'exec'), namespace)
torch.manual_seed(1999)
readout = namespace['MaskLocalAdjacencyReadout'](256)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    q = torch.randn(2, 8, 256)
    masks = torch.randn(2, 8, 16, requires_grad=True)
    pad = torch.zeros(2, 16, dtype=torch.bool)
    pad[:, -3:] = True
    mass = torch.rand(2, 16) + 1
    visual = torch.randn(2, 16, 256, requires_grad=True)
    xyz = torch.randn(2, 16, 3)
    output = readout(q, masks, pad, mass, visual, xyz)
    repeat = readout(q, masks, pad, mass, visual, xyz)
    perm = torch.arange(15, -1, -1)
    joint = readout(q, masks[:, :, perm], pad[:, perm], mass[:, perm], visual[:, perm], xyz[:, perm])
    queries = readout(q.roll(1, 1), masks.roll(1, 1), pad, mass, visual, xyz).roll(-1, 1)
    wrong = readout(q, masks, pad, mass, visual, xyz.roll(1, 1))
    detached = readout(q, masks.detach(), pad, mass, visual, xyz)
    bad_m, bad_v, bad_mass, bad_xyz = (masks.detach().clone(), visual.detach().clone(), mass.clone(), xyz.clone())
    bad_m[:, :, -3:] = bad_v[:, -3:] = bad_mass[:, -3:] = bad_xyz[:, -3:] = 100000000.0
    padded = readout(q, bad_m, pad, bad_mass, bad_v, bad_xyz)
    readout.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(output.reshape(-1, 2), torch.arange(16) % 2).backward()
    gradients = {k: float(v.grad.norm()) for k, v in readout.named_parameters()}
    row = dict(seed=seed, repeat=float((output - repeat).abs().max()), permutation=float((output - joint).abs().max()), query_permutation=float((output - queries).abs().max()), wrong_edges=float((output - wrong).abs().max()), padding=float((output - padded).abs().max()), detach=float((output - detached).abs().max()), mask_gradient=float(masks.grad.norm()), feature_gradient=float(visual.grad.norm()), gradients=gradients)
    assert row['repeat'] == row['padding'] == row['detach'] == 0
    assert max(row['permutation'], row['query_permutation']) < 1e-05
    assert min(row['wrong_edges'], row['mask_gradient'], row['feature_gradient']) > 0
    assert len(gradients) == 8 and all((v > 0 for v in gradients.values()))
    assert all((torch.isfinite(v.grad).all() for v in readout.parameters()))
    rows.append(row)
torch.manual_seed(1999)
q = torch.randn(1, 2, 256)
visual = torch.randint(-8, 9, (1, 6, 256)).float() / 8
xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [16.0, 0.0, 0.0]]])
masks = torch.zeros(1, 2, 6)
mass = torch.ones(1, 6)
pad = torch.zeros(1, 6, dtype=torch.bool)
changed = visual.roll(2, 1)
a = readout(q, masks, pad, mass, visual, xyz)
b = readout(q, masks, pad, mass, changed, xyz)
delta = float((a - b).abs().max())
assert torch.equal(visual.mean(1), changed.mean(1))
assert delta > 0
ties = xyz.clone()
ties[:, 2] = ties[:, 1]
tp = torch.tensor([5, 1, 4, 2, 0, 3])
ta = readout(q, masks, pad, mass, visual, ties)
tb = readout(q, masks[:, :, tp], pad[:, tp], mass[:, tp], visual[:, tp], ties[:, tp])
tie_delta = float((ta - tb).abs().max())
assert tie_delta < 1e-05
result = dict(complete=True, id='AdaRES-Y36', cpu_cases=32, cases=rows, equal_mask_feature_marginals=True, equal_global_mask_coordinate_moments=True, different_adjacency_control_delta=delta, tied_position_permutation_max=tie_delta, added_parameters=sum((p.numel() for p in readout.parameters())), new_trainable_tensors=8, source_sha256=hashlib.sha256(source).hexdigest(), repeat_max=max((r['repeat'] for r in rows)), permutation_max=max((max(r['permutation'], r['query_permutation']) for r in rows)), wrong_edges_min=min((r['wrong_edges'] for r in rows)), minimum_gradient=min((v for r in rows for v in r['gradients'].values())), limitations=['Structural discrimination, not native incidence, necessity or efficacy.', 'Fixed nearest-distance ties; no neighborhood size or radius search.', 'Same coordinates and masks preserve Y31 global mask moments.', 'Upstream spatial attention remains; no global blindness or novelty claim.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
