"""Frozen Y14 capture and train-only reliability screening."""

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


TAU = 0.25
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ("gate", "state_control", "assignment_control")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardize(x):
    std = x.std(unbiased=False)
    if not torch.isfinite(std) or std == 0:
        raise ValueError("Constant or nonfinite margin")
    return (x - x.mean()) / std


def entropy(probabilities, mass):
    p = probabilities.clamp(1e-7, 1 - 1e-7)
    values = -(p * p.log() + (1 - p) * (1 - p).log())
    return (values * mass).sum(-1) / mass.sum()


def weighted_iou(left, right, mass):
    a, b = left.float(), right.float()
    intersection = (a * mass) @ b.T
    union = (a * mass).sum(-1)[:, None] + (b * mass).sum(-1)[None, :]
    return intersection / (union - intersection).clamp_min(1)


def features(similarity, zw, zb, hw, hb):
    return torch.stack((
        similarity, 1 - similarity, (zw - zb).abs(), hw, hb,
        torch.ones_like(zw)), dim=-1)


def capture_predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
                      gt_pmasks, gt_spmasks, fps_seed_sp):
    sp = superpoints.detach().cpu().long()
    masks = []
    for i, truth in enumerate(gt_pmasks):
        logits = out["indis"][i].detach().cpu()
        probs = out["masks"][i].sigmoid().detach().cpu()
        binary = probs > 0.5
        mass = torch.bincount(sp, minlength=probs.shape[-1]).float()
        target_mass = torch.bincount(
            sp, weights=truth.detach().cpu().gt(0.5).float(),
            minlength=probs.shape[-1])
        intersection = binary.float() @ target_mass
        union = binary.float() @ mass + target_mass.sum() - intersection
        masks.append({
            "scan_id": str(scan_ids[0]), "ann_id": str(ann_ids[i]),
            "object_ids": tuple(int(value) for value in object_idss[i]),
            "margins": logits[:, 1] - logits[:, 0],
            "native_index": int(logits.softmax(-1)[:, 1].argmax()),
            "probabilities": probs, "binary": binary, "mass": mass,
            "entropy": entropy(probs, mass),
            "ious": intersection / union.clamp_min(1),
        })
    return {"rows": masks}


def combine(w2, b4):
    if (w2["scan_id"], w2["object_ids"], w2["ann_id"]) != (
            b4["scan_id"], b4["object_ids"], b4["ann_id"]):
        raise ValueError("Expression identity differs across models")
    if not torch.equal(w2["mass"], b4["mass"]):
        raise ValueError("Superpoint masses differ across models")
    sim = weighted_iou(w2["binary"], b4["binary"], w2["mass"])
    left, right = linear_sum_assignment(-sim.numpy())
    count = w2["margins"].numel()
    if len(left) != count or b4["margins"].numel() != count:
        raise ValueError("Incomplete query assignment")
    assignment = torch.empty(count, dtype=torch.long)
    assignment[torch.as_tensor(left)] = torch.as_tensor(right)
    cycled = assignment.roll(-1)
    zw, zb = standardize(w2["margins"]), standardize(b4["margins"])
    indices = torch.arange(count)
    phi = features(sim[indices, assignment], zw, zb[assignment],
                   w2["entropy"], b4["entropy"][assignment])
    ctrl_phi = features(sim[indices, cycled], zw, zb[cycled],
                        w2["entropy"], b4["entropy"][cycled])
    return {
        "scan_id": w2["scan_id"], "ann_id": w2["ann_id"],
        "object_ids": w2["object_ids"],
        "zw": zw, "zb": zb[assignment], "phi": phi,
        "assignment_zb": zb[cycled], "assignment_phi": ctrl_phi,
        "ious": w2["ious"], "assignment": assignment,
        "native_index": w2["native_index"],
        "b4_native_index": b4["native_index"],
        "margin_index": int(zw.argmax()),
    }


def load_model(cls, cfg, path):
    model = cls(**cfg.model).cuda()
    loaded = torch.load(path, map_location="cpu")
    incompatible = model.load_state_dict(loaded.get("model", loaded), strict=False)
    allowed = {"text_encoder.embeddings.position_ids",
               "text_encoder.embeddings.token_type_ids"}
    if incompatible.missing_keys or set(incompatible.unexpected_keys) - allowed:
        raise ValueError(str(incompatible))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.predict_by_feat = types.MethodType(capture_predictor, model)
    return model


def scene_indices(scene_inputs, count):
    selected, seen = [], set()
    for index, expressions in enumerate(scene_inputs):
        scans = {row["scene_id"] for row in expressions}
        if len(scans) != 1:
            raise ValueError("Dataset item contains mixed scans")
        scan = next(iter(scans))
        if scan not in seen:
            seen.add(scan)
            selected.append(index)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError("Not enough distinct scans")
    return selected


def capture(args):
    import gorilla
    from ipdn.dataset import build_dataset
    from ipdn.model_paper5.adares_b4_query_competition import IPDN as B4
    from ipdn.model_paper5.adares_w2_pointmass_competition import IPDN as W2
    from ipdn.utils import get_root_logger

    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.w2_config)
    data_cfg = getattr(cfg.data, args.split).copy()
    data_cfg.training, data_cfg.aug, data_cfg.with_elastic = False, False, False
    data_cfg.lang_num_max = 32
    dataset = build_dataset(data_cfg, get_root_logger())
    bcfg = gorilla.Config.fromfile(args.b4_config)
    w2 = load_model(W2, cfg, args.w2_checkpoint)
    b4 = load_model(B4, bcfg, args.b4_checkpoint)
    rows, repeats, diagnostics = [], [], []
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    selected = scene_indices(dataset.scene_inputs, args.scenes)
    for scene, item_index in enumerate(selected):
        batch = dataset.collate_fn([dataset[item_index]])
        with torch.no_grad():
            a1 = w2(batch, mode="predict")["rows"]
            b1 = b4(batch, mode="predict")["rows"]
            a2 = w2(batch, mode="predict")["rows"]
            b2 = b4(batch, mode="predict")["rows"]
        if not (len(a1) == len(b1) == len(a2) == len(b2)):
            raise ValueError("Capture expression counts differ")
        for aa, bb, ar, br in zip(a1, b1, a2, b2):
            first, second = combine(aa, bb), combine(ar, br)
            first["scene_index"] = second["scene_index"] = scene
            diagnostics.append({
                "scan_id": first["scan_id"], "ann_id": first["ann_id"],
                "object_ids": first["object_ids"],
                "w2_native_mismatch": first["native_index"] != second["native_index"],
                "b4_native_mismatch": first["b4_native_index"] != second["b4_native_index"],
                "assignment_changed_queries": int(
                    (first["assignment"] != second["assignment"]).sum()),
                "feature_max_diff": float((first["phi"] - second["phi"]).abs().max()),
                "margin_max_diff": float((first["zw"] - second["zw"]).abs().max()),
            })
            rows.append(first)
            repeats.append(second)
        print(json.dumps({"scene": scene, "expressions": len(rows),
                          "elapsed_s": time.monotonic() - start}), flush=True)
    result = {
        "rows": rows, "repeats": repeats, "repeat_diagnostics": diagnostics,
        "split": args.split, "elapsed_s": time.monotonic() - start,
        "scenes": len({r["scan_id"] for r in rows}),
        "selected_dataset_indices": selected,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "dual_model_parameters": sum(p.numel() for p in w2.parameters())
        + sum(p.numel() for p in b4.parameters()),
        "sources": {key: {"path": getattr(args, key),
                         "sha256": sha256(getattr(args, key))}
                    for key in ("w2_config", "b4_config",
                                "w2_checkpoint", "b4_checkpoint")},
    }
    torch.save(result, args.output)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("rows", "repeats", "repeat_diagnostics")}),
          flush=True)


def arm_inputs(row, arm):
    if arm == "assignment_control":
        return row["assignment_phi"], row["assignment_zb"]
    if arm == "state_control":
        return row["phi"].roll(-1, 0), row["zb"]
    return row["phi"], row["zb"]


def scores(row, arm, weights):
    phi, zb = arm_inputs(row, arm)
    gates = torch.sigmoid(phi @ weights.T)
    return row["zw"] + (gates * zb[:, None]).mean(-1)


def fit_one(rows, arm):
    w = torch.nn.Parameter(torch.zeros(6))
    optimizer = torch.optim.AdamW([w], lr=0.0001, weight_decay=0.0005)
    records = []
    scene_counts = defaultdict(int)
    prepared = []
    for row in rows:
        ordered = row["ious"][:, None] > row["ious"][None, :]
        if ordered.any():
            scene_counts[row["scan_id"]] += 1
            prepared.append((row, ordered))
    if not prepared:
        raise ValueError("No ordered training pairs")
    first_grad = None
    start = time.monotonic()
    for epoch in range(30):
        optimizer.zero_grad()
        total = 0.0
        for row, ordered in prepared:
            phi, zb = arm_inputs(row, arm)
            c = row["zw"] + torch.sigmoid(phi @ w) * zb
            loss = F.softplus(-(c[:, None] - c[None, :])[ordered] / TAU).mean()
            loss = loss / (len(scene_counts) * scene_counts[row["scan_id"]])
            loss.backward()
            total += float(loss.detach())
        if not torch.isfinite(w.grad).all():
            raise ValueError("Nonfinite gate gradient")
        if epoch == 0:
            first_grad = w.grad.tolist()
        optimizer.step()
        records.append(total)
    return w.detach(), {
        "loss_first": records[0], "loss_last": records[-1],
        "epochs": 30, "first_grad": first_grad,
        "all_first_grad_finite_nonzero": all(
            math.isfinite(g) and g != 0 for g in first_grad),
        "elapsed_s": time.monotonic() - start,
    }


def evaluate_row(row, repeat, weights):
    output = {"scan_id": row["scan_id"], "ann_id": row["ann_id"],
              "object_ids": row["object_ids"],
              "scene_index": row["scene_index"],
              "w2_iou": float(row["ious"][row["native_index"]]),
              "w2_margin_iou": float(row["ious"][row["margin_index"]]),
              "y13_iou": float(row["ious"][(row["zw"] + row["zb"]).argmax()])}
    for arm in ARMS:
        index = int(scores(row, arm, weights[arm]).argmax())
        again = int(scores(repeat, arm, weights[arm]).argmax())
        output[arm + "_iou"] = float(row["ious"][index])
        output[arm + "_index"] = index
        output[arm + "_repeat_mismatch"] = index != again
    return output


def paired_summary(rows, true, control):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scan_id"]].append(row[true] - row[control])
    deltas = [statistics.fmean(values) for values in grouped.values()]
    pos, neg = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
    trials = pos + neg
    p = min(1.0, 2 * sum(math.comb(trials, k)
                        for k in range(min(pos, neg) + 1)) / 2**trials)
    return {"scene_delta": statistics.fmean(deltas), "positive": pos,
            "negative": neg, "ties": len(deltas) - trials, "sign_p": p}


def summarize(rows):
    keys = ("w2_iou", "w2_margin_iou", "y13_iou") + tuple(
        arm + "_iou" for arm in ARMS)
    return {
        "expressions": len(rows), "scenes": len({r["scan_id"] for r in rows}),
        "mean_iou": {k: statistics.fmean(r[k] for r in rows) for k in keys},
        "acc25": {k: statistics.fmean(r[k] >= 0.25 for r in rows) for k in keys},
        "acc50": {k: statistics.fmean(r[k] >= 0.5 for r in rows) for k in keys},
        "paired": {ctrl: paired_summary(rows, "gate_iou", ctrl + "_iou")
                   for ctrl in ("w2", "state_control", "assignment_control")},
        "repeat_mismatches": {arm: sum(r[arm + "_repeat_mismatch"] for r in rows)
                              for arm in ARMS},
    }


def fit(args):
    torch.set_num_threads(1)
    data = torch.load(args.train, map_location="cpu")
    if data["split"] != "train" or data["scenes"] != 64:
        raise ValueError("Fit requires exactly train64")
    scenes = sorted({r["scan_id"] for r in data["rows"]})
    fold_map = {scan: i % 4 for i, scan in enumerate(scenes)}
    weights = {arm: [] for arm in ARMS}
    training, oof = [], []
    for fold in range(4):
        subset = [r for r in data["rows"] if fold_map[r["scan_id"]] != fold]
        local = {}
        for arm in ARMS:
            w, diagnostic = fit_one(subset, arm)
            weights[arm].append(w)
            local[arm] = w[None, :]
            training.append({"fold": fold, "arm": arm, **diagnostic})
            print(json.dumps(training[-1]), flush=True)
        for row, repeat in zip(data["rows"], data["repeats"]):
            if fold_map[row["scan_id"]] == fold:
                oof.append(evaluate_row(row, repeat, local))
    frozen = {arm: torch.stack(values) for arm, values in weights.items()}
    metadata = {"fold_map": fold_map, "weights": frozen,
                "train_scan_ids": scenes, "train_sha256": sha256(args.train),
                "sources": data["sources"]}
    torch.save(metadata, args.weights)
    report = {"status": "train_only_fit", "oof": summarize(oof),
              "training": training, "oof_rows": oof,
              "weights_sha256": sha256(args.weights),
              "feature_rank": int(torch.linalg.matrix_rank(torch.cat(
                  [r["phi"] for r in data["rows"]], dim=0))),
              "gate_parameter_count": 6,
              "backward_scope": "cached native features; not native tools/train.py",
              "sources": data["sources"],
              "repeat_native": {
                  model: sum(r[model + "_native_mismatch"]
                             for r in data["repeat_diagnostics"])
                  for model in ("w2", "b4")},
              "assignment_repeat_changed_expressions": sum(
                  r["assignment_changed_queries"] > 0
                  for r in data["repeat_diagnostics"]),
              "capture_peak_reserved_gib": data["peak_reserved_gib"],
              "capture_elapsed_s": data["elapsed_s"]}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("oof_rows", "training", "sources")}), flush=True)


def validate(args):
    data = torch.load(args.val, map_location="cpu")
    frozen = torch.load(args.weights, map_location="cpu")
    if data["split"] != "val" or data["scenes"] != 32:
        raise ValueError("Validation requires exactly val32")
    val_scans = {r["scan_id"] for r in data["rows"]}
    if val_scans.intersection(frozen["train_scan_ids"]):
        raise ValueError("Train/val scan leakage")
    if data["sources"] != frozen["sources"]:
        raise ValueError("Config or checkpoint changed after train capture")
    rows = [evaluate_row(row, repeat, frozen["weights"])
            for row, repeat in zip(data["rows"], data["repeats"])]
    summary = summarize(rows)
    passed = (summary["paired"]["w2"]["scene_delta"] > 0
              and summary["paired"]["state_control"]["scene_delta"] > 0
              and summary["paired"]["w2"]["sign_p"] < 0.05
              and summary["repeat_mismatches"]["gate"] == 0)
    report = {"summary": summary, "frozen_rule_passed": passed,
              "rows": rows, "repeat_diagnostics": data["repeat_diagnostics"],
              "sources": data["sources"], "weights_sha256": sha256(args.weights),
              "peak_reserved_gib": data["peak_reserved_gib"],
              "capture_elapsed_s": data["elapsed_s"],
              "dual_model_parameters": data["dual_model_parameters"],
              "active_queries": int(data["rows"][0]["zw"].numel()),
              "metric_scope": "diagnostic val32 point IoU, not full ScanRefer mIoU"}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "frozen_rule_passed": passed}), flush=True)


def self_test(args):
    torch.manual_seed(1999)
    probabilities = torch.rand(7, 5)
    mass = torch.tensor([1, 2, 4, 1, 3]).float()
    points = torch.repeat_interleave(torch.arange(5), mass.long())
    binary = probabilities > 0.5
    exact = weighted_iou(binary, binary.flip(0), mass)
    direct = weighted_iou(binary[:, points], binary.flip(0)[:, points],
                          torch.ones(points.numel()))
    assert torch.allclose(exact, direct, atol=1e-7)
    assert torch.allclose(entropy(probabilities, mass),
                          entropy(probabilities[:, points],
                                  torch.ones(points.numel())), atol=1e-7)
    phi, zw, zb = torch.rand(7, 6), torch.rand(7), torch.rand(7)
    w = torch.rand(6, requires_grad=True)
    gate = (phi @ w).sigmoid()
    c = zw + gate * zb
    loss = F.softplus(-(c[0] - c[1]) / TAU)
    gradient, = torch.autograd.grad(loss, w)
    analytical = (-torch.sigmoid(-(c[0] - c[1]) / TAU) / TAU
                  * (zb[0] * gate[0] * (1-gate[0]) * phi[0]
                     - zb[1] * gate[1] * (1-gate[1]) * phi[1]))
    assert torch.allclose(gradient, analytical, atol=1e-6)
    mock = [[{"scene_id": scan}] for scan in ("a", "a", "b", "b", "c")]
    assert scene_indices(mock, 3) == [0, 2, 4]
    assert {i for i in range(64) if i % 4 == 1}.isdisjoint(
        {i for i in range(64) if i % 4 != 1})
    result = {"weighted_iou_matches_expanded_points": True,
              "weighted_entropy_matches_expanded_points": True,
              "analytical_gradient_matches_autograd": True,
              "distinct_scan_sampling": True,
              "whole_scene_fold_isolation": True,
              "max_gradient_error": float((gradient - analytical).abs().max()),
              "scope": "synthetic implementation checks; no efficacy claim"}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    for name in ("w2-config", "b4-config", "w2-checkpoint", "b4-checkpoint"):
        cap.add_argument("--" + name, required=True)
    cap.add_argument("--split", choices=("train", "val"), required=True)
    cap.add_argument("--scenes", type=int, required=True)
    cap.add_argument("--gpu-id", type=int, required=True)
    cap.add_argument("--output", required=True)
    train = sub.add_parser("fit")
    train.add_argument("--train", required=True)
    train.add_argument("--weights", required=True)
    train.add_argument("--output", required=True)
    val = sub.add_parser("validate")
    val.add_argument("--val", required=True)
    val.add_argument("--weights", required=True)
    val.add_argument("--output", required=True)
    test = sub.add_parser("self-test")
    test.add_argument("--output", required=True)
    args = parser.parse_args()
    {"capture": capture, "fit": fit, "validate": validate,
     "self-test": self_test}[args.command](args)


if __name__ == "__main__":
    main()
