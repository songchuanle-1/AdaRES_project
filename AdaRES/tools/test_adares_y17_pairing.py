"""CPU pairing and objective contracts without importing sparse ops."""

import importlib.util
import argparse
import itertools
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ipdn/model_paper5/adares_y17_target_equivalent_pairing"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    pairing, objective = load("pairing"), load("paired_objective")
    cases = 0
    for length in range(1, 9):
        for assignment in itertools.product((0, 1), repeat=length):
            rows = [{"scene_id": "scene", "object_id": obj,
                     "ann_id": index, "token": [str(index % 3)]}
                    for index, obj in enumerate(assignment)]
            chunks = [rows[i:i+4] for i in range(0, length, 4)]
            after = pairing.paired_chunks(chunks)
            assert sorted(map(pairing.identity, rows)) == sorted(
                pairing.identity(r) for chunk in after for r in chunk)
            assert list(map(len, chunks)) == list(map(len, after))
            cases += 1
    torch.manual_seed(1999)
    masks = torch.randn(8, 3, 5, requires_grad=True)
    indis = torch.randn(8, 3, 2, requires_grad=True)
    out = {"masks": masks, "indis": indis,
           "batch_mask": torch.zeros(8, 5, dtype=torch.bool)}
    mass = torch.arange(1, 6).float().expand(8, -1)
    objects = [[0], [0], [1], [1], [0], [0], [1], [1]]
    tokens = torch.arange(8)[:, None]
    pairs = pairing.encoded_pairs(objects, tokens, [4, 4])
    value, count = objective.target_pair_js(out, pairs, objects, [4, 4], mass)
    assert count == 4 and torch.isfinite(value) and value > 0
    gradients = torch.autograd.grad(value, (masks, indis), retain_graph=True)
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
    zero, count = objective.target_pair_js(out, [[], []], objects, [4, 4], mass)
    assert count == 0 and float(zero) == 0 and zero.requires_grad
    assert all(torch.equal(g, torch.zeros_like(g))
               for g in torch.autograd.grad(zero, (masks, indis), retain_graph=True))
    identical_tokens = tokens.clone()
    identical_tokens[1] = identical_tokens[0]
    assert (0, 1) not in pairing.encoded_pairs(objects, identical_tokens, [4, 4])[0]
    failures = []
    for label, invalid in (
            ("cross_scene", [[(0, 4)], []]),
            ("wrong_object", [[(0, 2)], []]),
            ("self", [[(0, 0)], []]),
            ("duplicate", [[(0, 1), (0, 1)], []])):
        try:
            objective.target_pair_js(out, invalid, objects, [4, 4], mass)
        except ValueError:
            failures.append(label)
        else:
            raise AssertionError("Missing fail-closed: " + label)
    result = {"sampler_cases": cases, "pair_count": 4,
                      "objective": float(value),
                      "gradient_norms": [float(g.norm()) for g in gradients],
                      "no_pair_graph_zero": True,
                      "fail_closed": failures, "passed": True}
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
