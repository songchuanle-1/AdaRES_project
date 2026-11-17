"""CPU-only population coverage and analytic JS derivative checks."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def target(row):
    return tuple(sorted(int(v) for v in row.get("object_ids", [row.get("object_id")])))


def key(row):
    return (str(row["scene_id"]), target(row), str(row["ann_id"]))


def pairs(rows):
    return [(i, j) for i in range(len(rows)) for j in range(i+1, len(rows))
            if target(rows[i]) == target(rows[j])
            and rows[i]["token"] != rows[j]["token"]]


def js(z, w, mass):
    p, q = z.sigmoid(), w.sigmoid()
    m = (p+q)/2
    def kl(x):
        return x*(x/m).log()+(1-x)*((1-x)/(1-m)).log()
    return (((kl(p)+kl(q))/2)*mass).sum()/mass.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.annotation)
    rows = json.loads(path.read_text())
    scenes = defaultdict(list)
    for row in rows:
        scenes[str(row["scene_id"])].append(row)
    result = {"scope": "CPU prospective sampler, not native integration",
              "annotation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "rows": [], "complete": False}
    old_pairs = new_pairs = paired = chunks = singletons = 0
    for scan, original in sorted(scenes.items()):
        groups = defaultdict(list)
        for row in original:
            groups[target(row)].append(row)
        units, leftovers = [], []
        for ids, group in sorted(groups.items()):
            available = sorted(group, key=lambda r: str(r["ann_id"]))
            while available:
                left = available.pop(0)
                right = next((j for j, r in enumerate(available)
                              if r["token"] != left["token"]), None)
                if right is None:
                    leftovers.append(left)
                else:
                    units.append([left, available.pop(right)])
        ordered = [row for unit in units for row in unit] + leftovers
        if Counter(key(r) for r in original) != Counter(key(r) for r in ordered):
            raise AssertionError("Identity multiset differs: "+scan)
        old_chunks = [original[i:i+4] for i in range(0, len(original), 4)]
        new_chunks = [ordered[i:i+4] for i in range(0, len(ordered), 4)]
        if list(map(len, old_chunks)) != list(map(len, new_chunks)):
            raise AssertionError("Chunk sizes differ: "+scan)
        a = sum(len(pairs(chunk)) for chunk in old_chunks)
        b = sum(len(pairs(chunk)) for chunk in new_chunks)
        old_pairs += a
        new_pairs += b
        paired += 2*len(units)
        singletons += len(leftovers)
        chunks += len(new_chunks)
        result["rows"].append({
            "scan_id": scan, "expressions": len(original),
            "chunks": len(new_chunks), "native_same_target_pairs": a,
            "proposed_same_target_pairs": b, "paired_expressions": 2*len(units),
            "unpaired_expressions": len(leftovers),
            "new_chunks": [[{"identity": key(row), "token": row["token"]}
                            for row in chunk] for chunk in new_chunks]})
    torch.manual_seed(1999)
    z = torch.randn(64, dtype=torch.double, requires_grad=True)
    w = torch.randn(64, dtype=torch.double, requires_grad=True)
    mass = torch.arange(1, 65, dtype=torch.double)
    loss = js(z, w, mass)
    gz, gw = torch.autograd.grad(loss, (z, w))
    p, q = z.sigmoid(), w.sigmoid()
    m = (p+q)/2
    def analytic(x):
        return .5*(torch.logit(x)-torch.logit(m))*x*(1-x)*mass/mass.sum()
    errors = [float((gz-analytic(p)).abs().max()),
              float((gw-analytic(q)).abs().max())]
    if not torch.allclose(gz, analytic(p), rtol=1e-12, atol=1e-15):
        raise AssertionError("Left derivative mismatch")
    if not torch.allclose(gw, analytic(q), rtol=1e-12, atol=1e-15):
        raise AssertionError("Right derivative mismatch")
    self_loss = js(z, z, mass)
    self_grad = torch.autograd.grad(self_loss, z)[0]
    result["math"] = {
        "double_points": 64, "derivative_max_errors": errors,
        "self_js": float(self_loss), "self_gradient_max": float(self_grad.abs().max()),
        "symmetry_difference": float((loss-js(w, z, mass)).abs()),
        "finite": bool(torch.isfinite(gz).all() and torch.isfinite(gw).all())}
    result["summary"] = {
        "scans": len(scenes), "expressions": len(rows), "chunks": chunks,
        "paired_expressions": paired, "unpaired_expressions": singletons,
        "native_pair_comparisons": old_pairs, "proposed_pair_comparisons": new_pairs,
        "all_annotations_preserved": paired+singletons == len(rows),
        "original_per_scene_chunk_sizes_preserved": True}
    result["complete"] = True
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({"summary": result["summary"], "math": result["math"]}), flush=True)


if __name__ == "__main__":
    main()
