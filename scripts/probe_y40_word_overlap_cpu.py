"""Discriminate Y40 against the physical Y35 word/support realization."""

import ast
import hashlib
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(4)
assert not torch.cuda.is_available()


def extract(folder, name):
    path = Path("ipdn/model_paper5") / folder / "dec.py"
    source = path.read_bytes()
    tree = ast.parse(source.decode("utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name], hashlib.sha256(source).hexdigest(), tree


parent_cls, parent_hash, _ = extract(
    "adares_w2_pointmass_competition", "QueryCompetitionIndicator"
)
word_cls, word_hash, tree = extract(
    "adares_y35_language_evidence", "MaskLanguageEvidenceReadout"
)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    parent, word = parent_cls(256).eval(), word_cls(256).eval()
    query, masks = torch.randn(2, 8, 256), torch.randn(2, 8, 19)
    visual, words_a = torch.randn(2, 19, 256), torch.randn(2, 9, 256)
    padding, language_padding = torch.zeros(2, 19, dtype=torch.bool), torch.zeros(
        2, 9, dtype=torch.bool
    )
    padding[:, -3:], language_padding[:, -2:] = True, True
    mass = torch.rand(2, 19) + 1
    delta = torch.randn(2, 256)
    words_b = words_a.clone()
    words_b[:, 0] += delta
    words_b[:, 1] -= delta
    args = (query, masks, padding, mass, visual)
    left, right = word(*args, words_a, language_padding), word(
        *args, words_b, language_padding
    )
    repeat = word(*args, words_b, language_padding)
    permutation = torch.arange(8, -1, -1)
    permuted = word(*args, words_b[:, permutation], language_padding[:, permutation])
    mean_delta = (words_a[:, :7].mean(1) - words_b[:, :7].mean(1)).abs().max()
    parent_delta = (parent(query, masks, padding, mass) -
                    parent(query, masks, padding, mass)).abs().max()
    row = {
        "seed": seed,
        "token_mean_max": float(mean_delta),
        "same_query_mask_mass_visual": True,
        "w2_difference": float(parent_delta),
        "existing_y35_word_difference": float((left - right).abs().max()),
        "repeat_difference": float((right - repeat).abs().max()),
        "token_permutation_difference": float((right - permuted).abs().max()),
    }
    assert row["token_mean_max"] < 1e-6
    assert row["w2_difference"] == 0 and row["repeat_difference"] == 0
    assert row["existing_y35_word_difference"] > 0
    assert row["token_permutation_difference"] < 1e-5
    rows.append(row)
calls = [
    [arg.id for arg in n.args if isinstance(arg, ast.Name)]
    for n in ast.walk(tree)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    and n.func.attr == "language_readout"
]
assert any("inst_feats" in c and "lang_query" in c and "lang_masks" in c for c in calls)
result = {
    "complete": True, "id": "AdaRES-Y40", "cpu_cases": 32, "rows": rows,
    "parent_sha256": parent_hash, "existing_word_sha256": word_hash,
    "token_mean_max": max(r["token_mean_max"] for r in rows),
    "w2_max": max(r["w2_difference"] for r in rows),
    "existing_y35_min": min(r["existing_y35_word_difference"] for r in rows),
    "repeat_max": max(r["repeat_difference"] for r in rows),
    "permutation_max": max(r["token_permutation_difference"] for r in rows),
    "native_calls": calls,
    "conclusion": "Y35 already has the tested final-mask word/support alignment "
                  "path. The missing-realization premise for Y40 is disconfirmed; "
                  "do not re-admit this realization under a new name.",
    "limitations": [
        "Synthetic information-path coverage, not proof all language routes are equivalent.",
        "W2 upstream language encoding is not held blind in native training.",
        "No trained efficacy, semantic necessity, or novelty assertion.",
    ],
}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
