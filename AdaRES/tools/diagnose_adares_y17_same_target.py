"""Frozen same-target real-description native repeat and mismatch controls."""

import argparse
import hashlib
import json
import statistics
import time
import types
from pathlib import Path

import gorilla
import torch

from diagnose_adares_y14_reliability import capture_predictor, load_model
from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_w2_pointmass_competition import IPDN
from ipdn.utils import get_root_logger


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(row):
    return (row["scan_id"], row["object_ids"], row["ann_id"])


def pair(a, b):
    if not torch.equal(a["mass"], b["mass"]):
        raise ValueError("Spatial support mismatch")
    ia, ib = a["native_index"], b["native_index"]
    x, y, mass = a["binary"][ia], b["binary"][ib], a["mass"]
    overlap = float(((x & y).float() * mass).sum() /
                    ((x | y).float() * mass).sum().clamp_min(1))
    p = a["probabilities"][ia].double().clamp(1e-12, 1 - 1e-12)
    q = b["probabilities"][ib].double().clamp(1e-12, 1 - 1e-12)
    m = (p + q) / 2
    kl = lambda v: v * (v / m).log() + (1-v) * ((1-v) / (1-m)).log()
    js = float(((kl(p)+kl(q)) * .5 * mass).sum() / mass.sum())
    return {"mask_iou": overlap, "js": js}


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "checkpoint", "output", "manifest"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = cfg.data.train.copy()
    data_cfg.training = data_cfg.aug = data_cfg.with_elastic = False
    dataset = build_dataset(data_cfg, get_root_logger())
    scans = {}
    for chunk in dataset.scene_inputs:
        for row in chunk:
            scan = str(row["scene_id"])
            ids = tuple(sorted(int(v) for v in row.get(
                "object_ids", [row.get("object_id")])))
            text = tuple(row["token"])
            scans.setdefault(scan, {}).setdefault(ids, {}).setdefault(text, row)
    cohort = []
    for scan in sorted(scans):
        eligible = [(ids, list(texts.values())) for ids, texts
                    in sorted(scans[scan].items()) if len(texts) >= 2]
        if len(eligible) < 2:
            continue
        group = eligible[0][1][:2] + eligible[1][1][:2]
        cohort.append(group)
        if len(cohort) == 16:
            break
    if len(cohort) != 16:
        raise ValueError("Insufficient independent paired train scans")
    manifest = {"scope": "train-only, prediction-blind cohort",
                "scans": [[{"scan_id": r["scene_id"],
                            "object_ids": r.get("object_ids", [r.get("object_id")]),
                            "ann_id": r["ann_id"], "token": r["token"]}
                           for r in group] for group in cohort]}
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    model = load_model(IPDN, cfg, args.checkpoint)
    model.predict_by_feat = types.MethodType(capture_predictor, model)
    result = {"scope": "train16 real same-target annotations, not paraphrases",
              "manifest_sha256": sha(args.manifest),
              "checkpoint_sha256": sha(args.checkpoint),
              "config_sha256": sha(args.config), "rows": [], "complete": False}
    start = time.monotonic()
    for group in cohort:
        dataset.scene_inputs = [group]
        batch = dataset.collate_fn([dataset[0]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        if len(first) != 4 or [identity(r) for r in first] != [
                identity(r) for r in second]:
            raise ValueError("Native identity alignment failed")
        same = [pair(first[0], first[1]), pair(first[2], first[3])]
        wrong = [pair(first[0], first[2]), pair(first[1], first[3])]
        repeat = [pair(a, b) for a, b in zip(first, second)]
        row = {"scan_id": first[0]["scan_id"], "same": same, "wrong": wrong,
               "repeat": repeat,
               "identities": [identity(r) for r in first],
               "selection_repeat_flips": sum(a["native_index"] != b["native_index"]
                                             for a, b in zip(first, second)),
               "selected_iou": [float(r["ious"][r["native_index"]]) for r in first],
               "oracle_iou": [float(r["ious"].max()) for r in first],
               "oracle_pool_gains": []}
        sp = batch["superpoints"].cpu().long()
        for a, b in ((0, 1), (1, 0), (2, 3), (3, 2)):
            target = torch.bincount(sp, weights=batch["gt_pmasks"][a].cpu().gt(.5).float(),
                                    minlength=first[a]["mass"].numel())
            binary = first[b]["binary"].float()
            inter = binary @ target
            union = binary @ first[a]["mass"] + target.sum() - inter
            other_oracle = float((inter / union.clamp_min(1)).max())
            row["oracle_pool_gains"].append(max(
                other_oracle, row["oracle_iou"][a])-row["oracle_iou"][a])
        result["rows"].append(row)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps({"scan": row["scan_id"], "same": same,
                          "repeat": repeat}), flush=True)
    result["summary"] = {
        key + "_" + metric: statistics.fmean(
            statistics.fmean(p[metric] for p in row[key]) for row in result["rows"])
        for key in ("same", "wrong", "repeat") for metric in ("mask_iou", "js")}
    result["summary"]["same_pool_oracle_gain"] = statistics.fmean(
        statistics.fmean(r["oracle_pool_gains"]) for r in result["rows"])
    result["elapsed_s"] = time.monotonic() - start
    result["complete"] = True
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
