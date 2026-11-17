"""Test whether the proposed missing history path already exists physically."""

import ast
import hashlib
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(4)
assert not torch.cuda.is_available()


def actual(folder, name):
    path = Path("ipdn/model_paper5") / folder / "dec.py"
    data = path.read_bytes()
    tree = ast.parse(data.decode("utf-8"))
    node = next(item for item in tree.body
                if isinstance(item, ast.ClassDef) and item.name == name)
    namespace = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name], hashlib.sha256(data).hexdigest(), tree


parent_class, parent_sha, parent_tree = actual(
    "adares_w2_pointmass_competition", "QueryCompetitionIndicator")
history_class, history_sha, history_tree = actual(
    "adares_y30_query_trajectory", "QueryMaskTrajectoryReadout")
torch.manual_seed(1999)
parent, history = parent_class(256), history_class(256)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    query = torch.randn(2, 8, 256)
    final_masks = torch.randn(2, 8, 19)
    padding = torch.zeros(2, 19, dtype=torch.bool)
    padding[:, -3:] = True
    mass = torch.rand(2, 19) + 1
    queries = [query.clone() for _ in range(7)]
    left_masks = [final_masks.clone() for _ in range(7)]
    right_masks = [torch.randn_like(final_masks) for _ in range(6)] + [final_masks.clone()]
    parent_left = parent(queries[-1], left_masks[-1], padding, mass)
    parent_right = parent(queries[-1], right_masks[-1], padding, mass)
    left = history(queries, left_masks, padding, mass)
    right = history(queries, right_masks, padding, mass)
    repeat = history(queries, right_masks, padding, mass)
    permutation = torch.arange(18, -1, -1)
    permuted = history(queries, [item[:, :, permutation] for item in right_masks],
                       padding[:, permutation], mass[:, permutation])
    row = dict(
        seed=seed, final_masks_equal=bool(torch.equal(left_masks[-1], right_masks[-1])),
        final_queries_equal=True, w2_difference=float((parent_left-parent_right).abs().max()),
        existing_y30_history_difference=float((left-right).abs().max()),
        repeat_difference=float((right-repeat).abs().max()),
        permutation_difference=float((right-permuted).abs().max()),
        finite=bool(torch.isfinite(right).all()),
    )
    assert row["final_masks_equal"] and row["w2_difference"] == 0
    assert row["existing_y30_history_difference"] > 0 and row["repeat_difference"] == 0
    assert row["permutation_difference"] < 1e-5 and row["finite"]
    rows.append(row)
history_calls = [
    [arg.id for arg in node.args if isinstance(arg, ast.Name)]
    for node in ast.walk(history_tree)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    and node.func.attr == "trajectory_readout"
]
assert any("query_history" in call and "prediction_masks" in call for call in history_calls)
result = dict(
    complete=True, id="AdaRES-Y38", cpu_cases=len(rows), rows=rows,
    w2_max=max(row["w2_difference"] for row in rows),
    existing_y30_min=min(row["existing_y30_history_difference"] for row in rows),
    repeat_max=max(row["repeat_difference"] for row in rows),
    permutation_max=max(row["permutation_difference"] for row in rows),
    parent_sha256=parent_sha, existing_history_sha256=history_sha,
    existing_native_calls=history_calls,
    conclusion="The proposed missing explicit point-mass mask-history path is "
    "already implemented in Y30 and responds to the same frozen intervention. "
    "Do not re-admit that realization under Y38.",
    limitations=["This is source/CPU information-path coverage, not proof all history "
                 "models are equivalent or scientifically impossible.",
                 "Historical Y30 trained harm evidence remains unchanged; no renamed rerun.",
                 "No native efficacy or literature novelty claim."],
)
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
