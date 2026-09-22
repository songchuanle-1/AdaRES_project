from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.model_paper5.adares_c2_hard_negative_margin.loss import Criterion


def make_inputs():
    logits = torch.tensor(
        [[[-1.0, 0.0], [-1.0, 2.0], [-1.0, -1.0]]],
        requires_grad=True,
    )
    masks = torch.full((1, 3, 6), -4.0, requires_grad=True)
    masks.data[0, 0, :4] = 4.0
    masks.data[0, 1, :2] = 4.0
    masks.data[0, 2, :] = -4.0
    targets = [{"masks": torch.tensor([[1.0] * 4 + [0.0] * 2])}]
    return logits, masks, targets


def main():
    criterion = Criterion(
        hard_negative_weight=1.0,
        hard_negative_margin=0.1,
        hard_negative_iou_scale=1.0,
        hard_negative_min_gap=0.05,
    )
    logits, masks, targets = make_inputs()
    loss = criterion.loss_hard_negative(logits, [masks[0]], targets)
    assert torch.isfinite(loss)
    assert loss > 0
    loss.backward()
    assert logits.grad is not None, "missing logits gradient"
    assert torch.isfinite(logits.grad).all(), "invalid logits gradient"
    assert logits.grad.abs().sum() > 0, "zero logits gradient"
    # The oracle IoU defines the margin target, so it is intentionally detached.
    assert masks.grad is None, "mask IoU must remain a stop-gradient target"

    corrected = logits.detach().clone()
    corrected.data[0, 0, 0] = -1.0
    corrected.data[0, 0, 1] = 3.0
    corrected.data[0, 1, 0] = -1.0
    corrected.data[0, 1, 1] = 1.0
    corrected_loss = criterion.loss_hard_negative(
        corrected, [masks[0].detach()], targets
    )
    assert corrected_loss <= loss.detach()
    print(
        {
            "hard_negative_loss": float(loss.detach()),
            "corrected_loss": float(corrected_loss.detach()),
        }
    )


if __name__ == "__main__":
    main()
