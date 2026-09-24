from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.model_paper5.adares_d1_full_query_comparison.dec import (
    QueryCompetitionIndicator,
)


def main():
    torch.manual_seed(1999)
    module = QueryCompetitionIndicator(d_model=8)
    query = torch.randn(2, 4, 8, requires_grad=True)
    masks = torch.randn(2, 4, 6, requires_grad=True)
    padding = torch.tensor(
        [[False, False, False, False, True, True], [False] * 6]
    )

    logits = module(query, masks, padding)
    assert logits.shape == (2, 4, 2)
    assert torch.isfinite(logits).all()

    changed_padding = masks.detach().clone()
    changed_padding[0, :, 4:] = 1e4
    torch.testing.assert_close(
        logits[0], module(query, changed_padding, padding)[0]
    )

    changed_rival = query.detach().clone()
    changed_rival[0, 1] += torch.arange(8, dtype=query.dtype)
    assert not torch.allclose(
        logits[0, 0], module(changed_rival, masks, padding)[0, 0]
    )

    loss = logits.square().mean()
    loss.backward()
    for name, tensor in (("query", query), ("masks", masks)):
        assert tensor.grad is not None, f"missing {name} gradient"
        assert torch.isfinite(tensor.grad).all(), f"invalid {name} gradient"
        assert tensor.grad.abs().sum() > 0, f"zero {name} gradient"

    print(
        {
            "shape": list(logits.shape),
            "loss": float(loss.detach()),
            "query_grad": float(query.grad.norm()),
            "mask_grad": float(masks.grad.norm()),
        }
    )


if __name__ == "__main__":
    main()
