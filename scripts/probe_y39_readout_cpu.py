"""Test the actual frozen readout without CUDA or dataset outcomes."""
import ast
import hashlib
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(4)
assert not torch.cuda.is_available()
source_path = Path("ipdn/model_paper5/adares_y39_point_distribution/dec.py")
source = source_path.read_bytes()
node = next(item for item in ast.parse(source.decode("utf-8")).body
            if isinstance(item, ast.ClassDef) and item.name == "MaskPointDistributionReadout")
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    readout = namespace["MaskPointDistributionReadout"](256, 32)
    q = torch.randn(2, 8, 256, requires_grad=True)
    masks = torch.randn(2, 8, 10, requires_grad=True)
    padding = torch.zeros(2, 10, dtype=torch.bool)
    padding[0, 8:] = True
    mass = torch.randint(1, 30, (2, 10)).float()
    variance = torch.rand(2, 10, 32, requires_grad=True)
    output = readout(q, masks, padding, mass, variance)
    zero = readout(q, masks, padding, mass, torch.zeros_like(variance))
    wrong = readout(q, masks, padding, mass, variance.roll(1, 1))
    wrong_masks = readout(q, masks.roll(1, 1), padding, mass, variance)
    repeated = readout(q, masks, padding, mass, variance)
    sp = torch.arange(9, -1, -1)
    permuted = readout(q, masks[:, :, sp], padding[:, sp], mass[:, sp], variance[:, sp])
    poisoned = variance.detach().clone()
    poisoned[padding] = 1e5
    poison = readout(q, masks, padding, mass, poisoned)
    output.square().sum().backward()
    gradients = {name: float(value.grad.norm()) for name, value in readout.named_parameters()}
    row = dict(seed=seed, added_tensors=len(gradients), gradients=gradients,
               variance_input_grad=float(variance.grad.norm()),
               mask_grad_is_none=masks.grad is None,
               repeat_max=float((output - repeated).abs().max()),
               permutation_max=float((output - permuted).abs().max()),
               padding_poison_max=float((output - poison).abs().max()),
               zero_variance_delta=float((output - zero).abs().max()),
               wrong_variance_delta=float((output - wrong).abs().max()),
               wrong_mask_delta=float((output - wrong_masks).abs().max()))
    assert len(gradients) == 8 and all(value > 0 for value in gradients.values())
    assert all(torch.isfinite(value.grad).all() for value in readout.parameters())
    assert row["mask_grad_is_none"] and row["variance_input_grad"] > 0
    assert row["repeat_max"] == row["padding_poison_max"] == 0
    assert row["permutation_max"] < 1e-5
    assert min(row["zero_variance_delta"], row["wrong_variance_delta"], row["wrong_mask_delta"]) > 0
    rows.append(row)
result = dict(complete=True, id="AdaRES-Y39", cpu_only=True, rows=rows,
              source_sha256=hashlib.sha256(source).hexdigest(),
              fixtures=len(rows), minimum_gradient=min(v for row in rows for v in row["gradients"].values()),
              permutation_max=max(row["permutation_max"] for row in rows),
              repeat_max=max(row["repeat_max"] for row in rows),
              padding_poison_max=max(row["padding_poison_max"] for row in rows),
              wrong_variance_min=min(row["wrong_variance_delta"] for row in rows),
              wrong_mask_min=min(row["wrong_mask_delta"] for row in rows),
              limitations=["Synthetic graph and controls only; not native backward, utility or trained efficacy."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
