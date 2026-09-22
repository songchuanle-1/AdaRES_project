from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.model_paper5.adares_c1_listwise_expected_mask.loss import Criterion


def make_inputs():
    logits = torch.tensor(
        [[[-1.0, 1.0], [-1.5, 2.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    masks = torch.full((1, 3, 6), -4.0, requires_grad=True)
    masks.data[0, 0, :4] = 4.0
    masks.data[0, 1, :2] = 4.0
    targets = [{"masks": torch.tensor([[1.0] * 4 + [0.0] * 2])}]
    return logits, masks, targets


def main():
    criterion = Criterion(
        rank_weight=1.0,
        expected_mask_weight=0.5,
        rank_temperature=0.1,
        target_temperature=0.05,
    )
    logits, masks, targets = make_inputs()
    rank_loss = criterion.loss_listwise(logits, [masks[0]], targets)
    expected_loss = criterion.loss_expected_mask(
        logits, [masks[0]], targets
    )
    loss = rank_loss + expected_loss
    assert torch.isfinite(loss)
    loss.backward()
    for name, tensor in (("logits", logits), ("masks", masks)):
        assert tensor.grad is not None, f"missing {name} gradient"
        assert torch.isfinite(tensor.grad).all(), f"invalid {name} gradient"
        assert tensor.grad.abs().sum() > 0, f"zero {name} gradient"

    ordered = logits.detach().clone()
    ordered.data[0, :, 0] = torch.tensor([-1.0, -2.0, 0.0])
    ordered.data[0, :, 1] = torch.tensor([3.0, 2.0, 0.0])
    ordered_loss = criterion.loss_listwise(
        ordered, [masks[0].detach()], targets
    )
    assert ordered_loss < rank_loss.detach()
    print(
        {
            "rank_loss": float(rank_loss.detach()),
            "expected_mask_loss": float(expected_loss.detach()),
            "ordered_rank_loss": float(ordered_loss.detach()),
        }
    )


if __name__ == "__main__":
    main()
