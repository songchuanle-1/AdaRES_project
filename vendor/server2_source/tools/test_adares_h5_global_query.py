from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.model_paper5.adares_h5_late_global_query.dec import DEC


def main():
    torch.manual_seed(1999)
    model = DEC(num_layer=2, d_model=16, nhead=4, hidden_dim=32, lang_att=True)
    query = torch.randn(1, 4, 16, requires_grad=True)
    changed = query.detach().clone()
    changed[0, 3] += 5
    model.eval()
    output = model.global_query_attention(query)
    alternate = model.global_query_attention(changed)
    assert not torch.allclose(output[0, 0], alternate[0, 0])
    output[0, 0].square().sum().backward()
    assert query.grad[0, 3].abs().sum() > 0
    weight = model.global_query_attention.attn.in_proj_weight
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert weight.grad.abs().sum() > 0
    print("distant query communication and global path gradients: pass")


if __name__ == "__main__":
    main()
