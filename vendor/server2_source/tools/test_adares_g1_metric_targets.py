from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.model_paper5.adares_g1_metric_targets.loss import Criterion


def main():
    criterion = Criterion()
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    masks = [torch.tensor([[8., -8., -8.], [-8., 8., -8.], [-8., -8., 8.]])]
    target = [{"masks": torch.tensor([[1., 0., 0.]])}]
    matched = [(torch.tensor([1]), torch.tensor([0]))]
    loss = criterion.loss_indi(logits, masks, target, matched, torch.tensor([1.]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 1] < 0
    assert logits.grad[0, 1, 1] > 0
    assert logits.grad[0, 2, 1] > 0

    changed = [masks[0].clone().requires_grad_()]
    loss = criterion.loss_indi(logits.detach(), changed, target, matched, torch.tensor([1.]))
    assert not loss.requires_grad
    assert changed[0].grad is None
    print("metric targets, mismatched seed label, and detached mask targets: pass")


if __name__ == "__main__":
    main()
