"""Frozen TRAIN24 diagnosis using the actual production numeric AST."""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import zlib

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "1cb15474bfd346dcbc34837f3e9b79d3"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 180
capture_path = ROOT / ("pipeline/analyses/y49_train24_capture_v2_1cb1_" + TOKEN + ".json")
capture_sha = hashlib.sha256(capture_path.read_bytes()).hexdigest()
assert capture_sha == "277a11859224110feec8819de03f3d1aa3ae6bc37074d42f9fa5b29b4e842c11"
capture = json.loads(capture_path.read_text())
assert capture["complete"] and capture["native_scenes"] == 24 and capture["native_expressions"] == 96
assert capture["cohort_sha256"] == "3cd2f322a628613b4156e55a33b60b26b35bb458801de713937989e3e3e15336"
source_path = ROOT / "AdaRES/ipdn/model_paper5/adares_y49_resolvent_coherence/dec.py"
source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
assert source_sha == "50cfd1e821c188ddcb44bb929a6ffb84a186f0af89e1deb002aae10ca7e3bf31"
names = ("reference_inverse", "inverse_diagonal", "descriptors")
nodes = [n for n in ast.parse(source_path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(nodes) == 3
# Only adapt the final tensor container; all numeric function ASTs are unchanged.
torch = SimpleNamespace(tensor=lambda rows, dtype: np.array(rows, dtype=dtype), double=np.float64)
namespace = dict(globals(), forest_counts=dict(forest_calls=0, cyclic_fallback_calls=0))
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
production_inverse = namespace["inverse_diagonal"]
profile = []


def measured_inverse(graph, block_size=64):
    start = time.monotonic()
    value = production_inverse(graph, block_size)
    elapsed = time.monotonic() - start
    count, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels, minlength=count)
    edges = np.bincount(labels, weights=np.diff(graph.indptr), minlength=count) / 2
    cyclic = edges != counts - 1
    profile.append(dict(nodes=len(value), components=count,
                        cyclic_components=int(cyclic.sum()),
                        cyclic_nodes=int(counts[cyclic].sum()),
                        largest_component_nodes=int(counts.max()), seconds=elapsed))
    return value


namespace["inverse_diagonal"] = measured_inverse


def unpack(item):
    raw = zlib.decompress(base64.b64decode(item["zlib_base64"]))
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    return np.frombuffer(raw, dtype=np.dtype(item["dtype"])).reshape(item["shape"]).copy()


rows = []
start = time.monotonic()
for scene in capture["rows"]:
    for expression in scene["expressions"]:
        arrays = {k: unpack(v) for k, v in expression["packed_native_inputs"].items()}
        masses = arrays["masses"].astype(np.float64)
        masks = arrays["masks"] > 0
        edges = arrays["graph_edges"]
        graph = csr_matrix((np.ones(len(edges), bool), (edges[:, 0], edges[:, 1])), shape=(len(masses), len(masses)))
        profile_start = len(profile)
        values, partitions, unique = namespace["descriptors"](graph, masks, masses)
        assert values.shape == (128, 2) and np.isfinite(values).all()
        groups = {}
        for index, key in enumerate(partitions):
            groups.setdefault(key, []).append(index)
        witnesses = []
        for indices in groups.values():
            if len(indices) < 2:
                continue
            error = float(np.abs(values[indices] - values[indices[0]]).max())
            bound = 8 * np.finfo(np.float64).eps * max(1, len(masses)) * max(1.0, float(np.abs(values[indices]).max()), float(np.abs(values[indices[0]]).max()))
            if error <= bound:
                continue
            first = indices[0]
            contrasts = []
            for other in indices[1:]:
                difference = float(np.abs(values[first] - values[other]).max())
                if difference <= bound:
                    continue
                pooled = []
                for query in (first, other):
                    pooled.append(np.concatenate([np.average(arrays["tokens"][selected], axis=0, weights=masses[selected]) if selected.any() else np.zeros(arrays["tokens"].shape[-1]) for selected in (masks[query], ~masks[query])]))
                gt = arrays["TRAIN_gt_spmask"]
                ious = []
                for query in (first, other):
                    union = masses[masks[query] | gt].sum()
                    ious.append(float(masses[masks[query] & gt].sum() / union) if union else 0.0)
                contrasts.append(dict(query_ids=[first, other], descriptor_max_difference=difference,
                                      query_feature_max_difference=float(np.abs(arrays["query"][first] - arrays["query"][other]).max()),
                                      FG_BG_pooled_feature_max_difference=float(np.abs(pooled[0] - pooled[1]).max()),
                                      TRAIN_mask_pointmass_IoUs=ious, TRAIN_IoU_abs_gap=abs(ious[0] - ious[1])))
            witnesses.append(dict(query_indices=indices, descriptor_max_difference=error, existing8eps_nodes_bound=bound, contrasts=contrasts))
        row_profile = profile[profile_start:]
        row = dict(scan_id=scene["scan_id"], dataset_index=scene["dataset_index"], ann_id=expression["ann_id"],
                   nodes=len(masses), unique_selected_sets=unique, equal_mass_groups=sum(len(v) > 1 for v in groups.values()),
                   witnesses=witnesses, descriptor_ranges=np.ptp(values, axis=0).tolist(),
                   inverse_calls=len(row_profile), cyclic_calls=sum(p["cyclic_components"] > 0 for p in row_profile),
                   inverse_seconds=sum(p["seconds"] for p in row_profile))
        rows.append(row)
        print("Y49_TRAIN24_DIAG", len(rows), row["scan_id"], "witnesses", len(witnesses), "seconds", round(row["inverse_seconds"], 3), flush=True)

assert len(rows) == 96
cyclic = [p for p in profile if p["cyclic_components"]]
output = ROOT / ("pipeline/analyses/y49_train24_diagnosis_1cb1_" + TOKEN + ".json")
assert not output.exists()
result = dict(complete=True, id="AdaRES-Y49", observed_at_local=datetime.now().astimezone().isoformat(),
              capture_sha256=capture_sha, production_decoder_sha256=source_sha,
              numeric_AST_unchanged=True, return_container_only="numpy.float64 instead of torch.double, no numeric AST modification",
              native_scenes=24, native_expressions=96, old_TRAIN8_32_witness_count=0,
              new_expressions_with_witnesses=sum(bool(r["witnesses"]) for r in rows),
              new_witness_groups=sum(len(r["witnesses"]) for r in rows),
              total_seconds=time.monotonic() - start, inverse_calls=len(profile), cyclic_calls=len(cyclic),
              cyclic_inverse_seconds=sum(p["seconds"] for p in cyclic),
              forest_inverse_seconds=sum(p["seconds"] for p in profile if not p["cyclic_components"]),
              cyclic_profile=cyclic, rows=rows,
              limitations=["Prediction-blind frozen extension, no validation selection or optimizer updates.",
                           "A component-mass collision is not a collision of full parent query/pooled feature inputs.",
                           "TRAIN GT used only for passive per-witness diagnostic, never an inference input.",
                           "Profile is diagnostic CPU time, not production native resource admission or efficacy.",
                           "Original32-expression zero witness evidence and native runtime spikes remain binding."])
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "cyclic_profile")}, indent=2))
