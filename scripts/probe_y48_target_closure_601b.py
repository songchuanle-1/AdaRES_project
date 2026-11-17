"""Test final-target closure through the original native indicator loss."""

import ast
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import torch
from torch import nn

torch.set_num_threads(4)
loss_path = Path("ipdn/model_paper5/adares_w2_pointmass_competition/loss.py")
assert hashlib.sha256(loss_path.read_bytes()).hexdigest() == (
    "836ff790a0a8b9c55cf465a472c9462699f0753a7e87d0ef74a8946edf31565a")


def close_indices(masks, padding, indices):
    labels = torch.zeros(masks.shape[:2], device=masks.device, dtype=torch.bool)
    for b, (left, _) in enumerate(indices):
        labels[b, left] = True
    closed = labels.clone()
    mixed = 0
    for b in range(len(masks)):
        binary = masks[b, :, ~padding[b]].detach().gt(0)
        keys, inverse = torch.unique(binary, dim=0, return_inverse=True)
        counts = torch.bincount(inverse, minlength=len(keys))
        positives = torch.zeros_like(counts).scatter_add_(0, inverse, labels[b].long())
        nonempty = keys.any(-1)
        mixed += int(((positives > 0) & (positives < counts) & nonempty).sum())
        closed[b] |= (positives[inverse] > 0) & binary.any(-1)
        assert torch.equal(closed[b][~binary.any(-1)], labels[b][~binary.any(-1)])
    assert (closed | labels).equal(closed)
    replacement = []
    for value, (_, right) in zip(closed, indices):
        left = value.nonzero(as_tuple=True)[0]
        replacement.append((left, torch.zeros_like(left, dtype=right.dtype)))
    return replacement, labels, closed, mixed


def fixed_logits_controls(original, logits, targets, indices, num_boxes, masks, padding):
    replacement, labels, closed, mixed = close_indices(masks, padding, indices)
    left = logits.detach().clone().requires_grad_(True)
    right = logits.detach().clone().requires_grad_(True)
    parent = original(left, targets, indices, num_boxes)
    candidate = original(right, targets, replacement, num_boxes)
    pg = torch.autograd.grad(parent, left)[0]
    cg = torch.autograd.grad(candidate, right)[0]
    changed = closed & ~labels
    assert torch.isfinite(pg).all() and torch.isfinite(cg).all()
    if changed.any():
        assert (pg[..., 1][changed] >= 0).all()
        assert (cg[..., 1][changed] <= 0).all()
        assert ((pg - cg)[..., 1][changed] > 0).all()
    assert torch.equal(pg[~changed], cg[~changed])
    assert torch.equal(candidate, original(right, targets, replacement, num_boxes))
    return dict(
        mixed_nonempty_groups=mixed, changed_queries=int(changed.sum()),
        original_positive_queries=int(labels.sum()), closed_positive_queries=int(closed.sum()),
        empty_targets_unchanged=True, unchanged_query_gradients_exact=True,
        changed_foreground_gradient_direction_verified=True, cached_repeat_exact=True)


def cpu_mode():
    import yaml
    path = Path("ipdn/model_paper5/adares_w2_pointmass_competition/loss.py")
    criterion = next(node for node in ast.parse(path.read_text()).body
                     if isinstance(node, ast.ClassDef) and node.name == "Criterion")
    methods = {"__init__", "_get_src_permutation_idx", "loss_indi"}
    selected = ast.ClassDef(
        name="ActualCriterion",
        bases=[ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())],
        keywords=[], decorator_list=[],
        body=[node for node in criterion.body
              if isinstance(node, ast.FunctionDef) and node.name in methods])
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    namespace = dict(torch=torch, nn=nn, np=np)
    exec(compile(module, str(path), "exec"), namespace)
    cfg = yaml.safe_load(Path("configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml").read_text())
    actual = namespace["ActualCriterion"](**cfg["model"]["criterion"])
    assert actual.eos_coef == .1
    rows = []
    for seed in range(32):
        torch.manual_seed(1999 + seed)
        masks = torch.full((1, 128, 16), -1.)
        masks[:, :32, :8] = 1
        masks[:, 32:64, 8:] = 1
        masks[:, 96:] = 1
        padding = torch.zeros(1, 16, dtype=torch.bool)
        indices = [(torch.tensor([0, 40, 70]), torch.zeros(3, dtype=torch.long))]
        logits = torch.randn(1, 128, 2)
        num_boxes = torch.tensor(1.)
        row = fixed_logits_controls(
            actual.loss_indi, logits, [], indices, num_boxes, masks, padding)
        assert row["mixed_nonempty_groups"] == 2 and row["changed_queries"] == 62
        replacement, original, closed, _ = close_indices(masks, padding, indices)
        order = torch.randperm(128)
        inverse = torch.argsort(order)
        reordered = [(inverse[indices[0][0]], indices[0][1])]
        _, _, permuted, _ = close_indices(masks[:, order], padding, reordered)
        assert torch.equal(closed[:, order], permuted)
        row.update(seed=seed, query_permutation_exact=True, logits_are_constructed=True,
                   normalizer_is_unit_fixture_not_native_batch_count=True)
        rows.append(row)
    return dict(
        complete=True, id="AdaRES-Y48", CPU_cases=32, rows=rows,
        native_backward_executed=False,
        scientific_status="closure_target_CPU_controls_not_efficacy")


def native_mode():
    import gorilla
    from ipdn.dataset import build_dataset
    from ipdn.utils import get_root_logger
    from train import resolve_model_class

    assert P["physical_gpu"] in (0, 1) and torch.cuda.is_available()
    config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
    cfg = gorilla.Config.fromfile(config)
    gorilla.set_random_seed(1999)
    dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
    indices = [0, 44, 60, 101, 115, 126, 137, 171]
    assert len({dataset.scene_inputs[i][0]["scene_id"] for i in indices}) == 8
    model = resolve_model_class(config)(**cfg.model).cuda()
    checkpoint = torch.load(
        "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/"
        "adares_w2_pointmass_competition_res/20260926_145041/best.pth",
        map_location="cpu")
    loaded = model.load_state_dict(checkpoint["model"], strict=False)
    assert not loaded.missing_keys
    del checkpoint
    model.train()
    original = model.criterion.loss_indi
    state = {}
    buffers = {key: value.detach().cpu().clone() for key, value in model.named_buffers()}

    def before(module, inputs):
        state["prediction"] = inputs[0]
        state["calls"] = 0

    def instrumented(self, logits, targets, matched, num_boxes):
        state["calls"] += 1
        if state["calls"] != 1:
            return original(logits, targets, matched, num_boxes)
        pred = state["prediction"]
        assert torch.equal(logits.detach(), pred["indis"].detach())
        replacement, labels, closed, mixed = close_indices(
            pred["masks"], pred["batch_mask"], matched)
        state["first"] = (
            logits.detach().clone(), targets, matched, num_boxes.detach().clone(),
            pred["masks"].detach().clone(), pred["batch_mask"].detach().clone())
        return original(logits, targets, replacement if state["enabled"] else matched, num_boxes)

    handle = model.criterion.register_forward_pre_hook(before)
    model.criterion.loss_indi = types.MethodType(instrumented, model.criterion)
    rows = []
    try:
        for index in indices:
            gorilla.set_random_seed(1999)
            batch = dataset.collate_fn([dataset[index]])
            losses = {}
            for enabled in (False, True):
                model.zero_grad(set_to_none=True)
                for key, value in model.named_buffers():
                    value.copy_(buffers[key].to(value.device))
                gorilla.set_random_seed(1999)
                state.clear()
                state["enabled"] = enabled
                loss, _ = model(batch, mode="loss")
                assert torch.isfinite(loss)
                loss.backward()
                gradients = {
                    key: float(value.grad.norm())
                    for key, value in model.dec.indi_embedding.named_parameters()
                    if value.grad is not None}
                assert gradients and all(np.isfinite(v) for v in gradients.values())
                assert any(v > 0 for v in gradients.values())
                assert state["calls"] == model.dec.num_layer + 1
                losses["closed" if enabled else "parent"] = float(loss)
                if enabled:
                    controls = fixed_logits_controls(original, *state["first"])
                del loss
            rows.append(dict(
                dataset_index=index, scan_id=batch["scan_ids"][0], losses=losses,
                controls=controls, native_indicator_gradients=gradients,
                indicator_loss_calls=state["calls"],
                auxiliary_calls_use_unchanged_original=True))
            del batch
            state.clear()
            torch.cuda.empty_cache()
    finally:
        handle.remove()
        model.criterion.loss_indi = original
    assert sum(row["controls"]["changed_queries"] for row in rows) > 0
    return dict(
        complete=True, id="AdaRES-Y48", rows=rows, native_scenes=8,
        native_backward_executed=True,
        mixed_nonempty_groups=sum(row["controls"]["mixed_nonempty_groups"] for row in rows),
        changed_queries=sum(row["controls"]["changed_queries"] for row in rows),
        scientific_status="native_final_target_path_not_admission",
        limitations=[
            "Only original native final-indicator target indices changed; six auxiliary calls unchanged.",
            "Parent/closed losses are descriptive, not trained efficacy or model-selection criteria.",
            "No optimizer update, released-init control or independent production package yet.",
            "No full-loader bitwise repeat claim; fixed-logit controls are exact."])


result = cpu_mode() if P["physical_gpu"] is None else native_mode()
result["actual_loss_source_sha256"] = hashlib.sha256(loss_path.read_bytes()).hexdigest()
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
