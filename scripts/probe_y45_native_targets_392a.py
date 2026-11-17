"""Prediction-blind native TRAIN capture, point-exact targets and native CE."""

import hashlib
import json
from pathlib import Path
import types
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class

assert P["physical_gpu"] == 1
torch.set_num_threads(4)
gorilla.set_random_seed(1999)
config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
cfg = gorilla.Config.fromfile(config)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
indices, seen = [], set()
for index, chunk in enumerate(dataset.scene_inputs):
    if chunk[0]["scene_id"] not in seen:
        seen.add(chunk[0]["scene_id"])
        indices.append(index)
    if len(indices) == 8:
        break
assert indices == [0, 44, 60, 101, 115, 126, 137, 171]
checkpoint_path = Path("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth")
sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in Path("ipdn/model_paper5/adares_w2_pointmass_competition").glob("*.py")}
sources[config] = hashlib.sha256(Path(config).read_bytes()).hexdigest()
model = resolve_model_class(config)(**cfg.model).cuda()
checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
loaded = model.load_state_dict(checkpoint["model"], strict=False)
assert not loaded.missing_keys
assert not set(loaded.unexpected_keys) - {"text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
del checkpoint
model.train()
captured = {}
original_indi = model.criterion.loss_indi


def capture_indi(self, pred, targets, indices, num_boxes):
    captured.setdefault("indi", []).append(dict(
        logits=pred.detach().cpu().clone(),
        indices=[(left.detach().cpu().clone(), right.detach().cpu().clone()) for left, right in indices],
        num_boxes=num_boxes.detach().cpu().clone()))
    return original_indi(pred, targets, indices, num_boxes)


model.criterion.loss_indi = types.MethodType(capture_indi, model.criterion)
handle = model.criterion.register_forward_pre_hook(
    lambda module, inputs: captured.update(pred=inputs[0]))
rows = []
for index in indices:
    gorilla.set_random_seed(1999)
    batch = dataset.collate_fn([dataset[index]])
    assert len(batch["scan_ids"]) == 1
    captured.clear()
    with torch.no_grad():
        native_loss, native_losses = model(batch, mode="loss")
    pred = captured["pred"]
    masks = pred["masks"].detach().cpu()
    pad = pred["batch_mask"].detach().cpu()
    labels = batch["superpoints"].cpu().long()
    count = int(batch["batch_offsets"][-1])
    masses = model.point_masses(batch["superpoints"], batch["batch_offsets"], batch["scenes_len"]).cpu()
    truths = [v.detach().cpu().reshape(-1).gt(.5) for v in batch["gt_pmasks"]]
    assert len(truths) == masks.shape[0] == int(batch["scenes_len"][0])
    first = captured["indi"][0]
    assert first["logits"].shape == pred["indis"].shape
    indi_logits = first["logits"].cuda().requires_grad_()
    indi_indices = [(left.cuda(), right.cuda()) for left, right in first["indices"]]
    indi_loss = original_indi(indi_logits, [], indi_indices, first["num_boxes"].cuda())
    gradient = torch.autograd.grad(indi_loss, indi_logits)[0].detach().cpu()
    assert torch.isfinite(gradient).all() and float(gradient.norm()) > 0
    counts = torch.bincount(labels, minlength=count)
    expression_rows = []
    for b, truth in enumerate(truths):
        assert truth.numel() == labels.numel()
        valid = masks[b, :, ~pad[b]].gt(0)
        assert valid.shape == (128, count)
        assert torch.equal(masses.reshape(len(truths), count)[b].long(), counts)
        positive = torch.bincount(labels[truth], minlength=count)
        negative = counts - positive
        binary = valid.long()
        targets = torch.stack(((binary * positive).sum(-1),
                               (binary * negative).sum(-1),
                               ((1 - binary) * positive).sum(-1)), dim=-1)
        literal = valid[:, labels]
        expected = torch.stack(((literal & truth).sum(-1), (literal & ~truth).sum(-1),
                                (~literal & truth).sum(-1)), dim=-1)
        assert torch.equal(targets, expected)
        wrong_truth = truths[(b + 1) % len(truths)]
        wrong_pos = torch.bincount(labels[wrong_truth], minlength=count)
        wrong = torch.stack(((binary * wrong_pos).sum(-1),
                             (binary * (counts - wrong_pos)).sum(-1),
                             ((1 - binary) * wrong_pos).sum(-1)), dim=-1)
        reverse = torch.arange(labels.numel() - 1, -1, -1)
        permutation_target = torch.bincount(labels[reverse][truth[reverse]], minlength=count)
        assert torch.equal(permutation_target, positive)
        query_reverse = torch.arange(127, -1, -1)
        assert torch.equal(targets[query_reverse],
                           torch.stack(((binary[query_reverse] * positive).sum(-1),
                                        (binary[query_reverse] * negative).sum(-1),
                                        ((1 - binary[query_reverse]) * positive).sum(-1)), dim=-1))
        matched = torch.zeros(128, dtype=torch.bool)
        matched[first["indices"][b][0]] = True
        group_variation = {}
        for key, chosen in (("matched", matched), ("unmatched", ~matched)):
            subset = targets[chosen]
            group_variation[key] = dict(
                queries=int(chosen.sum()), distinct_confusions=len(torch.unique(subset, dim=0)),
                confusion_range=(subset.max(0).values - subset.min(0).values).tolist() if len(subset) else None)
        expression_rows.append(dict(
            ann_id=str(batch["ann_ids"][b]), object_ids=batch["object_idss"][b],
            truth_points=int(truth.sum()), queries=128, point_target_max_error=0,
            point_and_query_permutation_error=0, confusions=targets.tolist(),
            wrong_expression_target_gap=int((wrong - targets).abs().max()),
            wrong_expression_identical_truth=torch.equal(truth, wrong_truth),
            native_indicator_groups=group_variation,
            native_indicator_gradient_norm=float(gradient[b].norm())))
    rows.append(dict(dataset_index=index, scan_id=batch["scan_ids"][0],
                     points=len(labels), superpoints=count, expressions=expression_rows,
                     native_loss=float(native_loss), native_indi_loss=float(indi_loss),
                     existing_selector_gradient_norm=float(gradient.norm())))
handle.remove()
model.criterion.loss_indi = original_indi
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in sources.items())
result = dict(
    complete=True, id="AdaRES-Y45", rows=rows, manifest_indices=indices,
    scenes=len(rows), expressions=sum(len(r["expressions"]) for r in rows),
    native_queries=128, point_target_max_error=0,
    prediction_blind_sampling=True, train_config_sha256=sources[config],
    actual_parent_source_hashes=sources,
    nonidentical_wrong_expression_count=sum(not e["wrong_expression_identical_truth"] for r in rows for e in r["expressions"]),
    separated_wrong_expression_count=sum(e["wrong_expression_target_gap"] > 0 for r in rows for e in r["expressions"]),
    scientific_status="native_TRAIN_target_incidence_only_not_admitted",
    limitations=["Native W2 TRAIN masks, not synthetic fixtures; one frozen TRAIN8 cohort.",
                 "Gradients are existing matched-indicator logits, not a new learned decomposition.",
                 "Confusion variation within the same binary label is information availability, not trained utility.",
                 "GT counts are TRAIN targets only; no inference use, updates, admission or new model.",
                 "No fixed effect-size requirement, validation tuning or established novelty."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
