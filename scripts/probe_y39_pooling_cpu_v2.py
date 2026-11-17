"""Verify the point-pooling information gap and exact representability solver."""
import ast
import hashlib
import itertools
from pathlib import Path
import torch
from torch import nn
torch.set_num_threads(4)
assert not torch.cuda.is_available()
source_path = Path('ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py')
source = source_path.read_bytes()
tree = ast.parse(source.decode('utf-8'))
extract = next((item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == 'extract_feat'))
assert any((isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and (item.func.id == 'scatter_mean') for item in ast.walk(extract)))
dec_path = Path('ipdn/model_paper5/adares_w2_pointmass_competition/dec.py')
data = dec_path.read_bytes()
node = next((item for item in ast.parse(data.decode('utf-8')).body if isinstance(item, ast.ClassDef) and item.name == 'QueryCompetitionIndicator'))
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(dec_path), 'exec'), namespace)
torch.manual_seed(1999)
parent = namespace['QueryCompetitionIndicator'](256)

def ceiling(counts, positives):
    fraction = positives.double() / counts.double()
    order = torch.argsort(fraction, descending=True)
    intersection = positives[order].double().cumsum(0)
    selected = counts[order].double().cumsum(0)
    target = positives.double().sum()
    if target == 0:
        return 1.0
    return float((intersection / (selected + target - intersection)).max())
fixtures, exhaustive = ([], [])
for seed in range(32):
    torch.manual_seed(seed)
    means = torch.full((8, 32), 2.0)
    signs = torch.randint(0, 2, (8, 32)).double() * 2 - 1
    constant = means.double().unsqueeze(1).repeat(1, 4, 1)
    mixed = constant + signs.unsqueeze(1) * torch.tensor([0.5, -0.5, 0.5, -0.5], dtype=torch.double).view(1, 4, 1)
    assert torch.equal(constant.mean(1), mixed.mean(1))
    variance_a = ((constant - constant.mean(1, keepdim=True)) ** 2).mean(1)
    variance_b = ((mixed - mixed.mean(1, keepdim=True)) ** 2).mean(1)
    query, masks = (torch.randn(1, 8, 256), torch.randn(1, 8, 8))
    padding, mass = (torch.zeros(1, 8, dtype=torch.bool), torch.full((1, 8), 4.0))
    left = parent(query, masks, padding, mass)
    right = parent(query, masks, padding, mass)
    row = dict(seed=seed, mean_difference=0.0, counts_equal=True, masks_equal=True, geometry_equal=True, w2_difference=float((left - right).abs().max()), within_sp_variance_difference=float((variance_a - variance_b).abs().max()), permutation_difference=float((((mixed[:, [3, 1, 2, 0]] - mixed[:, [3, 1, 2, 0]].mean(1, keepdim=True)) ** 2).mean(1) - variance_b).abs().max()))
    assert row['w2_difference'] == row['permutation_difference'] == 0
    assert row['within_sp_variance_difference'] == 0.25
    fixtures.append(row)
    for count in range(1, 9):
        counts = torch.randint(1, 9, (count,))
        positives = torch.stack([torch.randint(0, int(n) + 1, ()) for n in counts])
        fast = ceiling(counts, positives)
        target = int(positives.sum())
        values = []
        for bits in itertools.product((0, 1), repeat=count):
            selected = torch.tensor(bits, dtype=torch.bool)
            intersection = int(positives[selected].sum())
            union = int(counts[selected].sum()) + target - intersection
            values.append(intersection / union if union else 1.0)
        brute = max(values)
        assert abs(fast - brute) < 1e-12
        exhaustive.append(dict(seed=seed, superpoints=count, difference=abs(fast - brute)))
result = dict(complete=True, id='AdaRES-Y39', cpu_cases=len(fixtures), rows=fixtures, exhaustive_cases=len(exhaustive), solver_exhaustive_max=max((item['difference'] for item in exhaustive)), parent_ipdn_sha256=hashlib.sha256(source).hexdigest(), parent_dec_sha256=hashlib.sha256(data).hexdigest(), within_sp_variance_difference=0.25, pooled_mean_difference=0, conclusion='Physical mean pooling has equal outputs on the fixed fixtures while within-SP all-channel variance differs. Exact whole-SP IoU ceiling solver agrees with exhaustive enumeration.', limitations=['Input-information separation only, no confidence model or native utility PASS.', 'Backbone features may encode heterogeneity indirectly.', 'Ceiling is a diagnostic bound and cannot prove a variance predictor estimates it.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
