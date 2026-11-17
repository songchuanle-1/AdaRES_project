from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.model_paper5.adares_b3_mask_evidence.dec import (
    MaskEvidenceIndicator,
)


def main():
    torch.manual_seed(1999)
    module = MaskEvidenceIndicator(d_model=8)
    query = torch.randn(2, 3, 8, requires_grad=True)
    masks = torch.randn(2, 3, 5, requires_grad=True)
    scene = torch.randn(2, 5, 8, requires_grad=True)
    padding = torch.tensor(
        [[False, False, False, True, True], [False] * 5]
    )

    logits = module(query, masks, scene, padding)
    assert logits.shape == (2, 3, 2)
    assert torch.isfinite(logits).all()

    changed_padding = scene.detach().clone()
    changed_padding[0, 3:] = 1e4
    changed = module(query, masks, changed_padding, padding)
    torch.testing.assert_close(logits[0], changed[0])

    loss = logits.square().mean()
    loss.backward()
    for name, tensor in (
        ("query", query),
        ("masks", masks),
        ("scene", scene),
    ):
        assert tensor.grad is not None, f"missing {name} gradient"
        assert torch.isfinite(tensor.grad).all(), f"invalid {name} gradient"
        assert tensor.grad.abs().sum() > 0, f"zero {name} gradient"

    print(
        {
            "shape": list(logits.shape),
            "loss": float(loss.detach()),
            "query_grad": float(query.grad.norm()),
            "mask_grad": float(masks.grad.norm()),
            "scene_grad": float(scene.grad.norm()),
        }
    )


if __name__ == "__main__":
    main()
