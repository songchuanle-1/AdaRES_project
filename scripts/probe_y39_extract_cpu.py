"""Actual extract_feat pooling algebra under point and voxel-map controls."""
import ast
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace
import torch
from torch_scatter import scatter_max, scatter_mean

torch.set_num_threads(4)
assert not torch.cuda.is_available()
path = Path("ipdn/model_paper5/adares_y39_point_distribution/ipdn.py")
source = path.read_bytes()
tree = ast.parse(source.decode("utf-8"))
node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "extract_feat")
scope = dict(scatter_max=scatter_max, scatter_mean=scatter_mean)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
module = SimpleNamespace(input_conv=lambda x: x, unet=lambda x: (x, None),
                         output_layer=lambda x: x, pool="mean")
module.extract_feat = MethodType(scope["extract_feat"], module)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    raw = torch.rand(48, 32)
    labels = torch.arange(8).repeat_interleave(6)
    voxel_order = torch.randperm(48)
    inverse = torch.argsort(voxel_order)
    x = SimpleNamespace(features=raw[voxel_order])
    pooled, variance = module.extract_feat(x, labels, inverse)
    mean = raw.reshape(8, 6, 32).mean(1)
    expected = ((raw.reshape(8, 6, 32) - mean[:, None]) ** 2).mean(1)
    point_order = torch.randperm(48)
    changed_mean, changed_variance = module.extract_feat(x, labels[point_order], inverse[point_order])
    loss = variance.sum()
    raw_grad = raw.clone().requires_grad_()
    _, reached = module.extract_feat(SimpleNamespace(features=raw_grad[voxel_order]), labels, inverse)
    reached.sum().backward()
    row = dict(seed=seed, mean_reference_max=float((pooled - mean).abs().max()),
               variance_reference_max=float((variance - expected).abs().max()),
               point_permutation_mean_max=float((pooled - changed_mean).abs().max()),
               point_permutation_variance_max=float((variance - changed_variance).abs().max()),
               raw_gradient_norm=float(raw_grad.grad.norm()),
               finite_gradient=bool(torch.isfinite(raw_grad.grad).all()))
    assert max(row[k] for k in row if k.endswith("_max")) < 1e-5
    assert row["raw_gradient_norm"] > 0 and row["finite_gradient"]
    rows.append(row)
result = dict(complete=True, id="AdaRES-Y39", cpu_only=True, rows=rows,
              cases=len(rows), source_sha256=hashlib.sha256(source).hexdigest(),
              algebra_max=max(max(row[k] for k in row if k.endswith("_max")) for row in rows),
              minimum_raw_gradient=min(row["raw_gradient_norm"] for row in rows),
              limitations=["Actual extract_feat source with identity backbone mocks; native voxel-to-point mapping and pooling algebra only.",
                           "Not a claim that sparse backbone or whole model is point-permutation bitwise invariant."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
