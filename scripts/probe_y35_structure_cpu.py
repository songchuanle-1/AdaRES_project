"""Actual-class CPU discrimination of visual-evidence-to-word attention."""

import ast
import hashlib
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(4)
assert not torch.cuda.is_available()
folder = Path("ipdn/model_paper5/adares_y35_language_evidence")
parent = folder.parent / "adares_w2_pointmass_competition"
source = (folder / "dec.py").read_bytes()
new = ast.parse(source.decode())
old = ast.parse((parent / "dec.py").read_text())
classes = {n.name: n for n in new.body if isinstance(n, ast.ClassDef)}
original = {n.name: n for n in old.body if isinstance(n, ast.ClassDef)}
assert all(ast.dump(classes[n]) == ast.dump(c)
           for n, c in original.items() if n != "DEC")
for name in ("__init__.py", "backbone.py", "ipdn.py", "loss.py", "sample_model.py"):
    assert (folder/name).read_bytes().replace(b"\r\n", b"\n") == (
        parent/name).read_bytes().replace(b"\r\n", b"\n")
config = Path("configs_song/model_paper5/adares_y35_language_evidence_res.yaml")
assert config.read_bytes().replace(b"\r\n", b"\n") == (
    config.parent/"adares_w2_pointmass_competition_res.yaml").read_bytes().replace(b"\r\n", b"\n")
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[classes["MaskLanguageEvidenceReadout"]],
                        type_ignores=[]), str(folder/"dec.py"), "exec"), namespace)
torch.manual_seed(1999)
readout = namespace["MaskLanguageEvidenceReadout"](256)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    q = torch.randn(2, 8, 256)
    masks = torch.randn(2, 8, 16, requires_grad=True)
    pad = torch.zeros(2, 16, dtype=torch.bool)
    pad[:, -3:] = True
    mass = torch.rand(2, 16) + 1
    visual = torch.randn(2, 16, 256, requires_grad=True)
    words = torch.randn(2, 10, 256, requires_grad=True)
    wp = torch.zeros(2, 10, dtype=torch.bool)
    wp[:, -2:] = True
    output = readout(q, masks, pad, mass, visual, words, wp)
    repeated = readout(q, masks, pad, mass, visual, words, wp)
    sp, lp = torch.arange(15, -1, -1), torch.arange(9, -1, -1)
    support = readout(q, masks[:, :, sp], pad[:, sp], mass[:, sp],
                      visual[:, sp], words, wp)
    language = readout(q, masks, pad, mass, visual, words[:, lp], wp[:, lp])
    queries = readout(q.roll(1, 1), masks.roll(1, 1), pad, mass,
                      visual, words, wp).roll(-1, 1)
    bad_m, bad_v, bad_w, bad_mass = (
        masks.detach().clone(), visual.detach().clone(),
        words.detach().clone(), mass.clone())
    bad_m[:, :, -3:] = bad_v[:, -3:] = bad_mass[:, -3:] = 1e8
    bad_w[:, -2:] = 1e8
    padded = readout(q, bad_m, pad, bad_mass, bad_v, bad_w, wp)
    wrong = readout(q, masks.roll(1, 1), pad, mass, visual, words, wp)
    detached = readout(q, masks.detach(), pad, mass, visual, words, wp)
    readout.zero_grad(set_to_none=True)
    F.cross_entropy(output.reshape(-1, 2), torch.arange(16) % 2).backward()
    gradients = {k: float(v.grad.norm()) for k, v in readout.named_parameters()}
    assert len(gradients) == 7 and all(v > 0 for v in gradients.values())
    assert all(torch.isfinite(v.grad).all() for v in readout.parameters())
    row = dict(seed=seed, repeat=float((output-repeated).abs().max()),
               padding=float((output-padded).abs().max()),
               support_permutation=float((output-support).abs().max()),
               word_permutation=float((output-language).abs().max()),
               query_permutation=float((output-queries).abs().max()),
               detach=float((output-detached).abs().max()),
               wrong_masks=float((output-wrong).abs().max()),
               mask_gradient=float(masks.grad.norm()),
               word_gradient=float(words.grad.norm()), gradients=gradients)
    assert row["repeat"] == row["padding"] == row["detach"] == 0
    assert max(row[k] for k in ("support_permutation", "word_permutation",
                               "query_permutation")) < 1e-5
    assert min(row["wrong_masks"], row["mask_gradient"], row["word_gradient"]) > 0
    rows.append(row)
words_a = torch.zeros(1, 4, 256)
words_a[0, 0, 0], words_a[0, 1, 0] = 1., -1.
words_a[0, 2, 2], words_a[0, 3, 2] = 1., -1.
words_b = words_a.clone()
words_b[0, :2] = 0
words_b[0, 0, 1], words_b[0, 1, 1] = 1., -1.
assert torch.equal(words_a.mean(1), words_b.mean(1))
q, masks, visual = torch.randn(1, 2, 256), torch.randn(1, 2, 8), torch.randn(1, 8, 256)
pad, wp, mass = torch.zeros(1, 8, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.bool), torch.ones(1, 8)
a = readout(q, masks, pad, mass, visual, words_a, wp)
b = readout(q, masks, pad, mass, visual, words_b, wp)
distribution_delta = float((a-b).abs().max())
assert distribution_delta > 0
result = dict(
    complete=True, id="AdaRES-Y35", cpu_cases=32, cases=rows,
    equal_language_first_moments=True, distribution_control_delta=distribution_delta,
    added_parameters=sum(p.numel() for p in readout.parameters()),
    new_trainable_tensors=7, source_sha256=hashlib.sha256(source).hexdigest(),
    repeat_max=max(r["repeat"] for r in rows),
    permutation_max=max(max(r[k] for k in ("support_permutation", "word_permutation",
                                           "query_permutation")) for r in rows),
    limitations=["Actual structural discrimination, not native incidence or efficacy.",
                 "Y28/Y30 rejections unchanged; upstream cross-attention exists.",
                 "All-padding language is not a valid native case; no novelty claim."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
