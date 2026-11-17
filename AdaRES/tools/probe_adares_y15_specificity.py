"""Isolate extra-query BCE gradients on frozen different-object train cohorts."""

import argparse
import gc
import json
from pathlib import Path

import gorilla
import torch
import torch.nn.functional as F

from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_y15_query_completion import IPDN
from ipdn.utils import get_root_logger
from train import load_shape_compatible_pretrain


def identity(row):
    return (str(row.get("scan_id", row.get("scene_id"))),
            tuple(int(x) for x in row.get("object_ids", [row.get("object_id")])),
            str(row["ann_id"]))


def gradvec(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True)
    return torch.cat([g.flatten() for g in grads])


def compare(a, b):
    return {"cosine": float((a @ b)/(a.norm()*b.norm()).clamp_min(1e-30)),
            "difference_norm": float((a-b).norm()),
            "max_difference": float((a-b).abs().max()),
            "left_norm": float(a.norm()), "right_norm": float(b.norm())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", required=True, type=int)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train, get_root_logger())
    source = json.loads(Path(args.source).read_text(encoding="utf-8"))["rows"]
    selected = []
    scans = []
    for row in source:
        if row["material"] and row["scan_id"] not in scans:
            scans.append(row["scan_id"])
            group = [row]
            objects = {identity(row)[1]}
            for peer in source:
                if peer["scan_id"] == row["scan_id"] and identity(peer)[1] not in objects:
                    group.append(peer)
                    objects.add(identity(peer)[1])
                    if len(group) == 4:
                        break
            selected.append(group)
            if len(scans) == 8:
                break
    annotations = {identity(row): row for chunk in dataset.scene_inputs
                   for row in chunk}
    frozen = [[annotations[identity(row)] for row in group] for group in selected]
    output = {"scope": "train-only frozen identities, current augmented targets",
              "cohort": [[list(identity(row)) for row in group]
                         for group in selected], "rows": [], "repeats": []}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    model = IPDN(**cfg.model).cuda()
    load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    params = [p for n, p in model.named_parameters()
              if n.startswith("dec.completion_query.")]
    for scene, group in enumerate(frozen):
        dataset.scene_inputs = [group]
        gorilla.set_random_seed(1999 + scene)
        batch = dataset.collate_fn([dataset[0]])
        raw_teacher = model.training_teacher_masks(batch)
        captured = []
        original = model.criterion.forward

        def capture(*inputs, **kwargs):
            captured.append((inputs, kwargs))
            return original(*inputs, **kwargs)

        model.criterion.forward = capture
        gorilla.set_random_seed(1999 + scene)
        loss, values = model.loss(**dict(batch, teacher_masks=raw_teacher))
        model.criterion.forward = original
        inputs, kwargs = captured[-1]
        prediction = inputs[0]
        objectives = []
        pads = prediction["batch_mask"]
        teacher_targets = kwargs["completion_targets"]
        material_rows = []
        oracle_targets = []
        for expr, teacher in enumerate(raw_teacher):
            sp = batch["superpoints"].to(teacher.device).long()
            mass = torch.bincount(sp, minlength=teacher.shape[-1]).float()
            target_mass = torch.bincount(
                sp, weights=batch["gt_pmasks"][expr].to(sp.device).gt(.5).float(),
                minlength=teacher.shape[-1])
            binary = teacher.gt(0).float()
            inter = binary @ target_mass
            union = binary @ mass + target_mass.sum()-inter
            oracle_targets.append(binary[(inter/union.clamp_min(1)).argmax()])
        for expr, metadata in enumerate(model.last_teacher_targets):
            if not metadata["material"]:
                continue
            peer = next((j for j, obj in enumerate(batch["object_idss"])
                         if tuple(obj) != tuple(batch["object_idss"][expr])), None)
            if peer is None:
                material_rows.append({"expression": expr, "control": "unavailable"})
                continue
            logits = prediction["masks"][expr, -1, ~pads[expr]]
            mass = prediction["point_mass"][expr, ~pads[expr]]
            gt = (inputs[4][expr] == batch["object_idss"][expr][0]).float()
            targets = [teacher_targets[expr], gt, oracle_targets[peer]]
            objectives = [(F.binary_cross_entropy_with_logits(
                logits, target.detach(), reduction="none")*mass).sum()/mass.sum()
                for target in targets]
            gradients = [gradvec(obj, params) for obj in objectives]
            material_rows.append({
                "scan_id": batch["scan_ids"][0],
                "object_ids": batch["object_idss"][expr],
                "ann_id": str(batch["ann_ids"][expr]),
                "peer_object_ids": batch["object_idss"][peer],
                "peer_ann_id": str(batch["ann_ids"][peer]),
                "teacher_oracle_iou": metadata["teacher_iou"],
                "student_oracle_iou": metadata["student_iou"],
                "teacher_gt_disagreement": float(
                    ((targets[0] != gt).float()*mass).sum()/mass.sum()),
                "teacher_wrong_disagreement": float(
                    ((targets[0] != targets[2]).float()*mass).sum()/mass.sum()),
                "teacher_vs_gt": compare(gradients[0], gradients[1]),
                "teacher_vs_wrong": compare(gradients[0], gradients[2]),
                "bce_losses": [float(obj) for obj in objectives],
                "finite_gradients": all(torch.isfinite(g).all() for g in gradients),
            })
        first_masks = prediction["masks"].detach().cpu()
        first_indis = prediction["indis"].detach().cpu()
        first_metadata = model.last_teacher_targets
        output["rows"].extend(material_rows)
        del loss, values, inputs, kwargs, prediction, captured, objectives
        if any("finite_gradients" in row for row in material_rows):
            del gradients, logits
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        captured = []
        model.criterion.forward = capture
        gorilla.set_random_seed(1999 + scene)
        with torch.no_grad():
            repeat_loss, repeat_values = model.loss(
                **dict(batch, teacher_masks=raw_teacher))
        model.criterion.forward = original
        second = captured[-1][0][0]
        second_masks = second["masks"].detach().cpu()
        second_indis = second["indis"].detach().cpu()
        output["repeats"].append({
            "scan_id": batch["scan_ids"][0],
            "raw_mask_max_difference": float((first_masks-second_masks).abs().max()),
            "raw_indicator_max_difference": float((first_indis-second_indis).abs().max()),
            "binary_flips": int((first_masks.gt(0) != second_masks.gt(0)).sum()),
            "selection_flips": int((first_indis.softmax(-1)[:, :, 1].argmax(-1)
                                   != second_indis.softmax(-1)[:, :, 1].argmax(-1)).sum()),
            "material_flags_equal": [r["material"] for r in first_metadata]
            == [r["material"] for r in model.last_teacher_targets]})
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps({"scan": scene, "material_rows": material_rows,
                          "repeat": output["repeats"][-1]}), flush=True)
        del second, captured, repeat_loss, repeat_values, raw_teacher, batch
    output["complete"] = True
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
