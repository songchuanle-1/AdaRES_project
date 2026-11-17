import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()
    import ipdn.model_paper5
    ipdn.model_paper5.__path__.append(str(args.package_dir.resolve()))
    module = importlib.import_module(
        "ipdn.model_paper5.adares_w2_pointmass_competition")
    model = module.IPDN.__new__(module.IPDN)
    torch.nn.Module.__init__(model)

    # Two concatenated scenes with two/three superpoints and unequal expression counts.
    points = torch.tensor([0, 0, 1, 2, 2, 2, 3, 3, 4])
    offsets = torch.tensor([0, 2, 5])
    actual = model.point_masses(points, offsets, [2, 1])
    expected = torch.tensor([2., 1., 2., 1., 3., 2., 1.])
    assert torch.equal(actual, expected), (actual.tolist(), expected.tolist())

    from ipdn.model_paper5.adares_w2_pointmass_competition.dec import (
        QueryCompetitionIndicator,
    )
    head = QueryCompetitionIndicator(4).eval()
    query = torch.randn(1, 3, 4)
    logits = torch.randn(1, 3, 5)
    padding = torch.tensor([[False, False, False, True, True]])
    mass = torch.tensor([[2., 1., 3., 0., 0.]])
    with torch.no_grad():
        original = head(query, logits, padding, mass)
        rescaled = head(query, logits, padding, mass * 7)
    assert torch.allclose(original, rescaled, atol=1e-5), (
        original.tolist(), rescaled.tolist())

    try:
        model.point_masses(torch.tensor([0, 2]), torch.tensor([0, 2]), [1])
        raise AssertionError("Noncontiguous indices should fail")
    except ValueError:
        pass
    print(json.dumps({
        "multiscene_expansion": actual.tolist(),
        "uniform_mass_scale_invariant": True,
        "bad_indices_rejected": True,
    }))


if __name__ == "__main__":
    main()
