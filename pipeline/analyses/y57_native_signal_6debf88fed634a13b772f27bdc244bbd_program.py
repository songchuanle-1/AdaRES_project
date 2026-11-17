"""Frozen descriptive follow-up; no learned production model or gate PASS."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

assert P["physical_gpu"] is None
INPUTS = {
    "Y56": ("/tmp/adares_y56_native_batch2_6debf88fed634a13b772f27bdc244bbd.json",
            "dff483ac71a5353b509289aaab76b3b3893cedeed29bfad35ee7601f0915a08d"),
    "Y57": ("/tmp/adares_y57_native_task_6debf88fed634a13b772f27bdc244bbd.json",
            "af0d796001a0b74174a103701a4f809d0605a5010955464a015d360ab5975846"),
}
data = {}
for key, (path, expected) in INPUTS.items():
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    data[key] = json.loads(raw)
    assert data[key]["complete"] and data[key]["expression_layer_records"] == 224
    assert data[key]["parameters_unchanged"] and data[key]["buffers_restored"]
    assert data[key]["optimizer_steps"] == 0 and not data[key]["weights_saved"]


def bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    assert len(values) == 8 and np.isfinite(values).all()
    indices = np.random.RandomState(1999).randint(0, len(values), (10000, len(values)))
    means = values[indices].mean(axis=1)
    return dict(mean=float(values.mean()),
                descriptive_scene_bootstrap95=np.quantile(means, [0.025, 0.975]).tolist())


def correlation(left, right):
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(spearmanr(left, right).correlation)
    assert np.isfinite(value)
    return value


def ols_predict(train_x, train_y, test_x):
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    train = np.column_stack((np.ones(len(train_x)), (train_x - mean) / scale))
    test = np.column_stack((np.ones(len(test_x)), (test_x - mean) / scale))
    weights, _, rank, _ = np.linalg.lstsq(train, train_y, rcond=None)
    return test @ weights, int(rank)


layers = []
incidence = []
for layer in range(7):
    rows = [row for row in data["Y57"]["rows"] if row["layer"] == layer]
    assert len(rows) == 32
    scenes = sorted({row["scene"] for row in rows})
    assert len(scenes) == 8
    per_expression = []
    arrays = {}
    for row in rows:
        confidence = np.asarray(row["confidence"], dtype=np.float64)
        quality = np.asarray(row["point_quality"], dtype=np.float64)
        dispersion = np.asarray(row["dispersion"], dtype=np.float64)
        assert confidence.shape == quality.shape == dispersion.shape == (128,)
        assert np.isfinite(confidence).all() and np.isfinite(quality).all()
        assert np.isfinite(dispersion).all() and (dispersion >= 0).all()
        assert ((confidence >= 0) & (confidence <= 1)).all()
        assert ((quality >= 0) & (quality <= 1)).all()
        arrays[(row["scene"], row["ann_id"])] = (
            confidence, quality, dispersion, np.roll(dispersion, 1))
        per_expression.append(dict(
            scene=row["scene"], ann_id=row["ann_id"],
            dispersion_error_rho=correlation(dispersion, np.abs(confidence - quality)),
            dispersion_quality_rho=correlation(dispersion, quality),
            rotated_label_error_rho=correlation(
                dispersion, np.abs(confidence - np.roll(quality, 1)))))
    folds = []
    for held_out in scenes:
        training = [values for (scene, _), values in arrays.items() if scene != held_out]
        testing = [values for (scene, _), values in arrays.items() if scene == held_out]
        train_c, train_y, train_d, train_rotated = [
            np.concatenate([values[index] for values in training]) for index in range(4)]
        test_c, test_y, test_d, test_rotated = [
            np.concatenate([values[index] for values in testing]) for index in range(4)]
        baseline, rank_baseline = ols_predict(train_c[:, None], train_y, test_c[:, None])
        alternative, rank_alternative = ols_predict(
            np.column_stack((train_c, train_d)), train_y,
            np.column_stack((test_c, test_d)))
        negative, rank_negative = ols_predict(
            np.column_stack((train_c, train_rotated)), train_y,
            np.column_stack((test_c, test_rotated)))
        baseline_mse = float(np.square(test_y - baseline).mean())
        alternative_mse = float(np.square(test_y - alternative).mean())
        negative_mse = float(np.square(test_y - negative).mean())
        folds.append(dict(
            scene=held_out, queries=int(len(test_y)), baseline_mse=baseline_mse,
            confidence_dispersion_mse=alternative_mse,
            confidence_rotated_dispersion_mse=negative_mse,
            gain_over_confidence=baseline_mse - alternative_mse,
            gain_over_matched_rotated_control=negative_mse - alternative_mse,
            design_ranks=[rank_baseline, rank_alternative, rank_negative]))
    associations = {}
    for name in ("dispersion_error_rho", "dispersion_quality_rho", "rotated_label_error_rho"):
        grouped = [[row[name] for row in per_expression
                    if row["scene"] == scene and row[name] is not None] for scene in scenes]
        if all(grouped):
            associations[name] = bootstrap([np.mean(values) for values in grouped])
        else:
            associations[name] = dict(valid_scenes=sum(bool(values) for values in grouped),
                                      result="undefined; never imputed as zero")
    layers.append(dict(
        layer=layer, primary=(layer == 6), scenes=8, expressions=32, queries=4096,
        associations=associations, folds=folds,
        loso_gain_over_confidence=bootstrap([row["gain_over_confidence"] for row in folds]),
        loso_gain_over_matched_control=bootstrap(
            [row["gain_over_matched_rotated_control"] for row in folds])))
    native_rows = [row for row in data["Y56"]["rows"] if row["layer"] == layer]
    assert len(native_rows) == 32
    incidence.append(dict(
        layer=layer, expressions=32,
        exact_joint_duplicate_excess=sum(row["joint_duplicate_excess"] for row in native_rows),
        hard_mask_duplicate_excess=sum(row["hard_mask_duplicate_excess"] for row in native_rows),
        minimum_query_unique=min(row["query_unique"] for row in native_rows),
        minimum_soft_mask_unique=min(row["soft_mask_unique"] for row in native_rows),
        minimum_hard_mask_unique=min(row["hard_mask_unique"] for row in native_rows)))

result = dict(
    complete=True, token=P["token"], inputs=INPUTS, primary_layer=6,
    primary=layers[6], exploratory_layers=layers[:6], native_incidence=incidence,
    units=dict(independent_scene_clusters=8, expressions=32,
               queries_per_expression=128, correlated_expression_layer_records=224),
    protocol="LOSO OLS: confidence only, confidence+dispersion, matched confidence+"
             "within-expression cyclic-shift1 dispersion. Train-fold standardization, "
             "unclipped prediction, no regularization/threshold/feature/layer search. "
             "Scene bootstrap10000/1999 descriptive only.",
    limits="Conditional only on confidence, NOT all native head inputs. Not causal "
           "identification, production model fitting, selector modification, "
           "pre-training efficacy gate, native-backward PASS, or population benefit.",
    production_model_or_config_changes=False, training_launches=0,
    weights_saved=False, native_backward=False, admission=False)
Path(P["output"]).write_text(json.dumps(result, indent=2), encoding="utf-8")
