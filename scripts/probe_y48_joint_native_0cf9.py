"""Joint native mask/ownership capture, with CPU-tested group bookkeeping."""

import hashlib
import json
from pathlib import Path
import types
import numpy as np
import torch

torch.set_num_threads(4)


def group_summary(binary, masses, positive, confusions=None):
    assert binary.dtype == bool and binary.shape[0] == len(positive)
    keys, inverse, frequencies = np.unique(np.packbits(binary, axis=1), axis=0,
                                           return_inverse=True, return_counts=True)
    groups = []
    for group in np.flatnonzero(frequencies > 1):
        indices = np.flatnonzero(inverse == group)
        assert (binary[indices] == binary[indices[0]]).all()
        mass = float(masses[binary[indices[0]]].sum())
        assert all(float(masses[binary[i]].sum()) == mass for i in indices)
        matched = positive[indices]
        item = dict(query_indices=indices.tolist(), size=len(indices), selected_point_mass=mass,
                    nonempty=mass > 0, positive_queries=int(matched.sum()),
                    negative_queries=int((~matched).sum()),
                    mixed_native_labels=bool(matched.any() and (~matched).any()))
        if confusions is not None:
            assert (confusions[indices] == confusions[indices[0]]).all()
            item["identical_literal_point_confusion"] = confusions[indices[0]].tolist()
        groups.append(item)
    return dict(
        groups=groups, nonempty_duplicate_groups=sum(g["nonempty"] for g in groups),
        mixed_nonempty_groups=sum(g["nonempty"] and g["mixed_native_labels"] for g in groups),
        mixed_empty_groups=sum(not g["nonempty"] and g["mixed_native_labels"] for g in groups))


def cpu_test():
    rows = []
    for seed in range(32):
        random = np.random.RandomState(1999 + seed)
        binary = np.zeros((128, 16), dtype=bool)
        binary[:32, :8] = True
        binary[32:64, 8:] = True
        binary[96:] = True
        positive = np.zeros(128, dtype=bool)
        positive[[0, 40, 70]] = True
        masses = random.randint(1, 17, 16).astype(np.float64)
        result = group_summary(binary, masses, positive)
        assert result["nonempty_duplicate_groups"] == 3
        assert result["mixed_nonempty_groups"] == 2 and result["mixed_empty_groups"] == 1
        qorder, porder = random.permutation(128), random.permutation(16)
        permuted = group_summary(binary[qorder][:, porder], masses[porder], positive[qorder])
        signatures = lambda x: sorted((g["size"], g["selected_point_mass"], g["positive_queries"],
                                       g["negative_queries"], g["mixed_native_labels"]) for g in x["groups"])
        assert signatures(result) == signatures(permuted)
        assert signatures(result) == signatures(group_summary(binary, masses, positive))
        rows.append(dict(seed=seed, correct_literal_counts=True, exact_joint_permutation=True,
                         exact_repeat=True))
    return dict(
        complete=True, id="AdaRES-Y48", rows=rows, CPU_cases=32,
        native_joint_capture_executed=False, actual_native_mixed_nonempty_groups=None,
        scientific_status="joint_capture_bookkeeping_CPU_test_not_native_conflict",
        limitations=["Synthetic label bookkeeping only; no native model, ownership conflict or benefit inferred.",
                     "The GPU branch below is compiled but unrun until a live-idle authorized exclusive GPU.",
                     "Empty groups are counted separately, not turned into positive-target support."])


def native_capture():
    import gorilla
    from ipdn.dataset import build_dataset
    from ipdn.utils import get_root_logger
    from train import resolve_model_class

    assert P["physical_gpu"] in (0, 1) and torch.cuda.is_available()
    gorilla.set_random_seed(1999)
    config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
    cfg = gorilla.Config.fromfile(config)
    dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
    indices = [0, 44, 60, 101, 115, 126, 137, 171]
    assert len({dataset.scene_inputs[i][0]["scene_id"] for i in indices}) == 8
    checkpoint_path = Path(
        "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/"
        "adares_w2_pointmass_competition_res/20260926_145041/best.pth")
    model = resolve_model_class(config)(**cfg.model).cuda()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    loaded = model.load_state_dict(checkpoint["model"], strict=False)
    assert not loaded.missing_keys
    del checkpoint
    model.train()
    captured = {}
    original = model.criterion.loss_indi

    def capture_indi(self, pred, targets, matches, num_boxes):
        captured.setdefault("indi", []).append(dict(
            logits=pred.detach().cpu().clone(),
            indices=[(left.detach().cpu().clone(), right.detach().cpu().clone())
                     for left, right in matches],
            num_boxes=num_boxes.detach().cpu().clone()))
        return original(pred, targets, matches, num_boxes)

    model.criterion.loss_indi = types.MethodType(capture_indi, model.criterion)
    handle = model.criterion.register_forward_pre_hook(
        lambda module, inputs: captured.update(pred=inputs[0]))
    rows = []
    try:
        for index in indices:
            gorilla.set_random_seed(1999)
            batch = dataset.collate_fn([dataset[index]])
            captured.clear()
            with torch.no_grad():
                loss, _ = model(batch, mode="loss")
            assert torch.isfinite(loss)
            pred = captured["pred"]
            masks, padding = pred["masks"].detach().cpu(), pred["batch_mask"].detach().cpu()
            first = captured["indi"][0]
            assert torch.equal(first["logits"], pred["indis"].detach().cpu())
            labels = batch["superpoints"].cpu().long()
            count = int(batch["batch_offsets"][-1])
            counts = torch.bincount(labels, minlength=count)
            masses = model.point_masses(batch["superpoints"], batch["batch_offsets"], batch["scenes_len"]).cpu()
            expressions = []
            for b, truth in enumerate(batch["gt_pmasks"]):
                truth = truth.detach().cpu().reshape(-1).gt(.5)
                assert truth.numel() == labels.numel()
                binary = masks[b, :, ~padding[b]].gt(0)
                assert binary.shape == (128, count)
                assert torch.equal(masses.reshape(len(masks), count)[b].long(), counts)
                positive_points = torch.bincount(labels[truth], minlength=count)
                values = binary.long()
                confusions = torch.stack(((values * positive_points).sum(-1),
                                          (values * (counts - positive_points)).sum(-1),
                                          ((1 - values) * positive_points).sum(-1)), -1)
                matched = np.zeros(128, dtype=bool)
                matched[first["indices"][b][0].numpy()] = True
                summary = group_summary(binary.numpy(), counts.numpy(), matched, confusions.numpy())
                expressions.append(dict(ann_id=str(batch["ann_ids"][b]),
                                        object_ids=batch["object_idss"][b],
                                        native_matching_indices=[v.tolist() for v in first["indices"][b]],
                                        exact_hard_mask_groups=summary))
            rows.append(dict(dataset_index=index, scan_id=batch["scan_ids"][0], expressions=expressions))
            del batch, pred, loss
            captured.clear()
            torch.cuda.empty_cache()
    finally:
        handle.remove()
        model.criterion.loss_indi = original
    return dict(
        complete=True, id="AdaRES-Y48", rows=rows, native_scenes=8,
        native_expressions=sum(len(r["expressions"]) for r in rows), queries=128,
        native_joint_capture_executed=True,
        actual_native_mixed_nonempty_groups=sum(e["exact_hard_mask_groups"]["mixed_nonempty_groups"]
                                               for r in rows for e in r["expressions"]),
        actual_native_mixed_empty_groups=sum(e["exact_hard_mask_groups"]["mixed_empty_groups"]
                                            for r in rows for e in r["expressions"]),
        actual_loss_source_sha256=hashlib.sha256(Path("ipdn/model_paper5/adares_w2_pointmass_competition/loss.py").read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        scientific_status="native_joint_ownership_mask_incidence_only",
        limitations=["Masks/labels from the SAME forward; no cross-capture index inference.",
                     "Native one-to-many ownership source unchanged; no new loss/model/weights or inference GT.",
                     "No optimizer update, production implementation, admission or population benefit.",
                     "No global bitwise input-loader/repeat trajectory claim."])


result = cpu_test() if P["physical_gpu"] is None else native_capture()
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
