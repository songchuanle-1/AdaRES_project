"""Append to the frozen native setup: full-input observational controls."""

import base64
import io
import zlib

previous_path = Path("/tmp/adares_y57_native_task_6debf88fed634a13b772f27bdc244bbd.json")
assert hashlib.sha256(previous_path.read_bytes()).hexdigest() == (
    "af0d796001a0b74174a103701a4f809d0605a5010955464a015d360ab5975846")
previous = json.loads(previous_path.read_text())
previous_rows = {(r["scene"], r["ann_id"]): r for r in previous["rows"] if r["layer"] == 6}
assert len(previous_rows) == 32
captured = []
started = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
try:
    for pair in range(0, 8, 2):
        items, identities = [], []
        for scene in FROZEN["scenes"][pair:pair + 2]:
            gorilla.set_random_seed(1999)
            item = dataset[scene["dataset_index"]]
            single = dataset.collate_fn([copy.deepcopy(item)])
            assert single["scan_ids"][0] == scene["scan_id"]
            assert [str(v) for v in single["ann_ids"]] == scene["ann_ids"]
            items.append(item)
            identities.extend((scene["scan_id"], ann) for ann in scene["ann_ids"])
        batch = dataset.collate_fn(copy.deepcopy(items))
        targets = pair_targets(batch, [
            row for scene in FROZEN["scenes"][pair:pair + 2]
            for row in FROZEN["labels"] if row["scene"] == scene["scan_id"]])
        calls = []

        def observe(module, inputs, output):
            layer = len(calls)
            calls.append(layer)
            if layer != 6:
                return
            query, masks, padding, mass = inputs
            assert query.shape == (8, 128, 256)
            for index, (scene, ann) in enumerate(identities):
                valid = ~padding[index]
                probabilities = masks[index, :, valid].sigmoid()
                total = targets[index]["total"].to(query.device, dtype=query.dtype)
                positive = targets[index]["positive"].to(query.device, dtype=query.dtype)
                assert torch.equal(mass[index, valid], total)
                weighted = probabilities * total.unsqueeze(0)
                areas = weighted.sum(-1)
                intersection = weighted @ probabilities.T
                overlap = intersection / (
                    areas[:, None] + areas[None, :] - intersection).clamp_min(1e-6)
                overlap.fill_diagonal_(0)
                competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
                normalized = module.query_norm(query[index])
                context = competition @ normalized
                variance = (competition @ normalized.square() - context.square()).clamp_min(0)
                dispersion = variance.square().sum(-1).sqrt()
                evidence = torch.cat((
                    normalized, context, normalized - context,
                    overlap.max(-1).values.unsqueeze(-1)), dim=-1)
                head_input = module.evidence_norm(evidence)
                reconstructed = module.head(head_input)
                error = float((reconstructed - output[index]).abs().max())
                bound = 64 * torch.finfo(query.dtype).eps * max(
                    1.0, float(output[index].abs().max()))
                assert error <= bound, "Full native head inputs were not reconstructed"
                binary = (probabilities >= 0.5).to(query.dtype)
                numerator = (binary * positive.unsqueeze(0)).sum(-1)
                quality = numerator / (
                    (binary * total.unsqueeze(0)).sum(-1) + positive.sum() - numerator + 1e-6)
                confidence = output[index].softmax(-1)[:, 1]
                old = previous_rows[(scene, ann)]
                assert old["GT_binding"] == targets[index]["metadata"]
                captured.append(dict(
                    scene=scene, ann_id=ann, binding=targets[index]["metadata"],
                    head_input=head_input.detach().cpu().numpy(),
                    confidence=confidence.detach().cpu().numpy(),
                    quality=quality.detach().cpu().numpy(),
                    dispersion=dispersion.detach().cpu().numpy(),
                    reconstruction_error=error, reconstruction_bound=bound,
                    previous_confidence_maxdiff=float(np.max(np.abs(
                        confidence.detach().cpu().numpy() - np.asarray(old["confidence"])))),
                    previous_quality_maxdiff=float(np.max(np.abs(
                        quality.detach().cpu().numpy() - np.asarray(old["point_quality"]))))))

        hook = model.dec.indi_embedding.register_forward_hook(observe)
        try:
            gorilla.set_random_seed(1999)
            with torch.no_grad():
                loss, _ = model(copy.deepcopy(batch), mode="loss")
            assert torch.isfinite(loss).all() and len(calls) == 7
        finally:
            hook.remove()
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        del loss, batch, items
finally:
    for name, value in model.named_buffers():
        value.copy_(buffers[name])
assert parameter_versions == {name: value._version for name, value in model.named_parameters()}
assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
assert len(captured) == 32
assert digest(checkpoint_path) == FROZEN["checkpoint_sha256"]
native_seconds = time.perf_counter() - started
peak_reserved = int(torch.cuda.max_memory_reserved())
model.cpu()
torch.cuda.empty_cache()


def predict(train_x, train_y, test_x):
    train_x, train_y, test_x = (
        np.asarray(value, dtype=np.float64) for value in (train_x, train_y, test_x))
    mean, scale = train_x.mean(0), train_x.std(0)
    scale = np.where(scale == 0, 1.0, scale)
    x, test = (train_x - mean) / scale, (test_x - mean) / scale
    center = train_y.mean()
    system = x.T @ x + np.eye(x.shape[1])
    beta = np.linalg.solve(system, x.T @ (train_y - center))
    prediction = test @ beta + center
    assert np.isfinite(prediction).all()
    return prediction


def bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.shape == (8,) and np.isfinite(values).all()
    indices = np.random.RandomState(1999).randint(0, 8, (10000, 8))
    return dict(mean=float(values.mean()),
                descriptive_scene_bootstrap95=np.quantile(
                    values[indices].mean(1), [0.025, 0.975]).tolist())


scenes = sorted({r["scene"] for r in captured})
assert len(scenes) == 8
for row in captured:
    row["baseline"] = np.column_stack((row["head_input"], row["confidence"]))
    row["alternative"] = np.column_stack((row["baseline"], row["dispersion"]))
    row["negative"] = np.column_stack((row["baseline"], np.roll(row["dispersion"], 1)))
    assert row["baseline"].shape == (128, 770)
    assert row["alternative"].shape == row["negative"].shape == (128, 771)
folds = []
for held_out in scenes:
    training = [r for r in captured if r["scene"] != held_out]
    testing = [r for r in captured if r["scene"] == held_out]
    train_y = np.concatenate([r["quality"] for r in training])
    test_y = np.concatenate([r["quality"] for r in testing])
    errors = {}
    for kind in ("baseline", "alternative", "negative"):
        fitted = predict(np.concatenate([r[kind] for r in training]), train_y,
                         np.concatenate([r[kind] for r in testing]))
        errors[kind] = float(np.square(test_y - fitted).mean())
    folds.append(dict(
        scene=held_out, queries=int(len(test_y)), mse=errors,
        gain_over_full_inputs=errors["baseline"] - errors["alternative"],
        gain_over_same_capacity_shifted=errors["negative"] - errors["alternative"]))

archive = io.BytesIO()
np.savez_compressed(archive,
                    head_input=np.stack([r["head_input"] for r in captured]),
                    confidence=np.stack([r["confidence"] for r in captured]),
                    quality=np.stack([r["quality"] for r in captured]),
                    dispersion=np.stack([r["dispersion"] for r in captured]))
packed = archive.getvalue()
result = dict(
    complete=True, id="AdaRES-Y57", primary_layer=6, scenes=scenes, expressions=32,
    native_TRAIN_batch2=True, actual_sparse_FPS=True,
    full_native_head_input_reconstruction=True,
    records=[{k: v for k, v in r.items() if k in (
        "scene", "ann_id", "binding", "reconstruction_error", "reconstruction_bound",
        "previous_confidence_maxdiff", "previous_quality_maxdiff")} for r in captured],
    fixed_ridge_penalty=1.0, folds=folds,
    loso_gain_over_full_inputs=bootstrap([r["gain_over_full_inputs"] for r in folds]),
    loso_gain_over_same_capacity_shifted=bootstrap(
        [r["gain_over_same_capacity_shifted"] for r in folds]),
    input_archive=dict(format="numpy savez_compressed", bytes=len(packed),
                       sha256=hashlib.sha256(packed).hexdigest(),
                       base64=base64.b64encode(packed).decode("ascii")),
    native_seconds=native_seconds, elapsed_seconds=time.perf_counter() - started,
    peak_reserved_bytes=peak_reserved, parameters_unchanged=True, buffers_restored=True,
    optimizer_steps=0, production_weights_saved=False, production_model_changes=False,
    limits="One fixed ridge function class on8TRAIN scene clusters; not conditional "
           "independence identification, causal task defect, trained model benefit, "
           "native-backward/resource/admission PASS or novelty. Negative folds retained.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
