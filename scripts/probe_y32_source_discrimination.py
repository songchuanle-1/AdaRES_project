"""Execute actual historical confidence losses before claiming a new pathway."""

import ast
import hashlib
from pathlib import Path
import types

import torch
import torch.nn.functional as F

torch.set_num_threads(4)
marker = Path(".codex_source_commit").read_text().strip()
assert marker == "5f2489de78196e8d89c004e452a665c0946f2a94"
paths = {
    "w2": "ipdn/model_paper5/adares_w2_pointmass_competition/loss.py",
    "g1": "ipdn/model_paper5/adares_g1_metric_targets/loss.py",
    "c1": "ipdn/model_paper5/adares_c1_listwise_expected_mask/loss.py",
    "y29_reference": "ipdn/model_paper5/adares_b0/loss.py",
}
sources, methods = {}, {}
for name, filename in paths.items():
    path = Path(filename)
    source = path.read_bytes()
    sources[name] = dict(path=filename, sha256=hashlib.sha256(source).hexdigest())
    tree = ast.parse(source.decode("utf-8"))
    wanted = {
        "w2": ["_get_src_permutation_idx", "loss_indi"],
        "g1": ["loss_indi"],
        "c1": ["_query_iou", "loss_listwise"],
        "y29_reference": ["loss_iou_ranking"],
    }[name]
    namespace = dict(torch=torch, F=F)
    nodes = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name in wanted]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    obj = types.SimpleNamespace(eos_coef=.1, rank_temperature=.1,
                                target_temperature=.05, rank_pair_margin=0.0)
    for wanted_name in wanted:
        function = namespace[wanted_name]
        setattr(obj, wanted_name, function if wanted_name == "_query_iou"
                else types.MethodType(function, obj))
    methods[name] = obj


def exact_iou(masks, truth, mass):
    binary = masks.sigmoid().ge(.5).double()
    truth = truth.gt(.5).double()
    intersection = (binary * truth * mass).sum(-1)
    union = ((binary + truth - binary * truth) * mass).sum(-1)
    return intersection / union.clamp_min(1)


def proposed(logits, quality, matched):
    if float(quality.sum()) > 0:
        target = quality / quality.sum()
    elif matched.numel():
        target = torch.zeros_like(quality)
        target[matched] = 1.0 / matched.numel()
    else:
        indices = [(matched, matched)]
        return methods["w2"].loss_indi(logits[None], [], indices,
                                      logits.new_tensor(1.0))
    return -(target.detach() * (logits[:, 1] - logits[:, 0]).log_softmax(0)).sum()


torch.manual_seed(1999)
rows = []
for case in range(32):
    q, n = 8, 17
    logits = torch.randn(q, 2, dtype=torch.float32, requires_grad=True)
    masks = torch.randn(q, n, dtype=torch.float64)
    truth = torch.arange(n).remainder(3).eq(0).double()
    matched = torch.tensor([0, 1], dtype=torch.long)
    if case == 0:
        masks[:] = -1
    elif case == 1:
        masks[:] = truth.mul(2).sub(1)
    elif case == 2:
        matched = matched[:0]
        masks[:] = -1
    targets = [dict(masks=truth[None])]
    indices = [(matched, torch.zeros_like(matched))]
    quality = exact_iou(masks, truth, torch.ones(n))
    actual_c1_quality = methods["c1"]._query_iou(masks, truth[None]).double()
    quality_error = float((quality - actual_c1_quality).abs().max())
    assert quality_error < 1e-7
    native = methods["w2"].loss_indi(logits[None], targets, indices,
                                    logits.new_tensor(1.0))
    candidate = proposed(logits, quality, matched)
    c1 = methods["c1"].loss_listwise(logits[None], [masks], targets)
    g1 = methods["g1"].loss_indi(logits[None], [masks], targets, indices,
                                logits.new_tensor(1.0))
    y29 = methods["y29_reference"].loss_iou_ranking(
        logits[None], [masks], truth[None], torch.ones(1, n, dtype=torch.bool)).sum()
    named = dict(w2=native, candidate=candidate, c1=c1, g1=g1, y29_reference=y29)
    grads = {k: torch.autograd.grad(v, logits, retain_graph=True)[0]
             if v.requires_grad else torch.zeros_like(logits) for k, v in named.items()}
    changed = masks.clone()
    changed[2] = truth.mul(2).sub(1)
    changed_quality = exact_iou(changed, truth, torch.ones(n))
    candidate2 = proposed(logits, changed_quality, matched)
    c12 = methods["c1"].loss_listwise(logits[None], [changed], targets)
    native2 = methods["w2"].loss_indi(logits[None], targets, indices,
                                     logits.new_tensor(1.0))
    candidate_target = quality / quality.sum() if float(quality.sum()) > 0 else None
    c1_target = actual_c1_quality.div(.05).softmax(0)
    ordering_identity = candidate_target is None or torch.equal(
        torch.sign(candidate_target[:, None] - candidate_target[None]),
        torch.sign(c1_target[:, None] - c1_target[None]))
    assert ordering_identity
    assert float((native - native2).abs()) == 0
    rows.append(dict(
        case=case, point_iou_vs_actual_c1_max=quality_error,
        c1_candidate_target_order_identical=ordering_identity,
        losses={k: float(v.detach()) for k, v in named.items()},
        gradient_max_vs_candidate={k: float((v - grads["candidate"]).abs().max())
                                   for k, v in grads.items()},
        nonmatched_quality_intervention=dict(
            w2_loss_delta=float((native2-native).detach()),
            candidate_loss_delta=float((candidate2-candidate).detach()),
            c1_loss_delta=float((c12-c1).detach())),
        finite=all(bool(torch.isfinite(v).all()) for v in grads.values())))
assert all(r["finite"] for r in rows)
result = dict(
    complete=True, id="AdaRES-Y32", cases=rows, case_count=32,
    sources=sources, source_marker=marker,
    all_equal_mass_point_quality_matches_actual_c1=True,
    all_c1_candidate_target_order_identical=True,
    w2_ignores_nonmatched_quality=True,
    c1_already_consumes_all_query_target_quality=True,
    candidate_is_not_numerically_identical_to_c1=True,
    candidate_and_c1_intervention_nonzero_cases=sum(
        abs(r["nonmatched_quality_intervention"]["candidate_loss_delta"]) > 0
        and abs(r["nonmatched_quality_intervention"]["c1_loss_delta"]) > 0 for r in rows),
    limitations=[
        "CPU synthetic/source audit, not TRAIN incidence or trained efficacy.",
        "Different target normalization/loss replacement is not numerical identity.",
        "All-query target-quality pathway already exists in actual C1.",
        "Unequal point measure and zero-target handling are not isolated efficacy.",
        "No training, GPU use, target/weight/temperature tuning, or old source edits."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
