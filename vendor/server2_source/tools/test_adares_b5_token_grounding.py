from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.model_paper5.adares_b5_token_grounding.dec import (
    TokenGroundingIndicator,
)


def main():
    torch.manual_seed(1999)
    module = TokenGroundingIndicator(d_model=8)
    query = torch.randn(2, 3, 8, requires_grad=True)
    tokens = torch.randn(2, 6, 8, requires_grad=True)
    padding = torch.tensor(
        [[False, False, False, False, True, True], [False] * 6]
    )

    logits = module(query, tokens, padding)
    assert logits.shape == (2, 3, 2)
    assert torch.isfinite(logits).all()

    changed_padding = tokens.detach().clone()
    changed_padding[0, 4:] = 1e4
    changed = module(query, changed_padding, padding)
    torch.testing.assert_close(logits[0], changed[0])

    changed_token = tokens.detach().clone()
    changed_token[0, 1] = changed_token[0, 1].roll(1)
    token_logits = module(query, changed_token, padding)
    assert not torch.allclose(logits[0], token_logits[0])

    loss = logits.square().mean()
    loss.backward()
    for name, tensor in (("query", query), ("tokens", tokens)):
        assert tensor.grad is not None, f"missing {name} gradient"
        assert torch.isfinite(tensor.grad).all(), f"invalid {name} gradient"
        assert tensor.grad.abs().sum() > 0, f"zero {name} gradient"

    print(
        {
            "shape": list(logits.shape),
            "loss": float(loss.detach()),
            "query_grad": float(query.grad.norm()),
            "token_grad": float(tokens.grad.norm()),
        }
    )


if __name__ == "__main__":
    main()
