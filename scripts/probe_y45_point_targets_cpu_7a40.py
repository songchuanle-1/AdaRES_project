"""Point-exact confusion-target mapping under prediction-blind CPU fixtures."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace
import torch

torch.set_num_threads(4)
root = Path(P["root"])
path = root / "ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py"
tree = ast.parse(path.read_text())
node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "point_masses")
namespace = dict(torch=torch)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
parent = SimpleNamespace()
parent.point_masses = MethodType(namespace["point_masses"], parent)


def point_targets(mask_binary, superpoints, truth, point_mass):
    assert mask_binary.dtype == torch.bool and truth.dtype == torch.bool
    positives = torch.bincount(superpoints[truth], minlength=point_mass.numel()).double()
    negative = point_mass.double() - positives
    mask = mask_binary.double()
    tp = (mask * positives).sum(-1)
    fp = (mask * negative).sum(-1)
    fn = ((1 - mask) * positives).sum(-1)
    return torch.stack((tp, fp, fn), dim=-1)


cases = []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    labels = torch.arange(11).repeat_interleave(torch.arange(1, 12))
    order = torch.randperm(labels.numel())
    labels = labels[order]
    offsets = torch.tensor([0, 11])
    counts = parent.point_masses(labels, offsets, [1])
    if isinstance(counts, (list, tuple)):
        counts = counts[0]
    counts = counts.reshape(-1)[:11]
    assert torch.equal(counts.double(), torch.bincount(labels, minlength=11).double())
    masks = torch.rand(128, 11) > 0.5
    masks[0] = False
    masks[1] = True
    truth = torch.arange(labels.numel()).remainder(3).eq(0)
    targets = point_targets(masks, labels, truth, counts)
    literal = masks[:, labels]
    expected = torch.stack(((literal & truth).sum(-1),
                            (literal & ~truth).sum(-1),
                            (~literal & truth).sum(-1)), dim=-1).double()
    assert torch.equal(targets, expected)
    joint = torch.randperm(labels.numel())
    assert torch.equal(targets, point_targets(masks, labels[joint], truth[joint], counts))
    query = torch.randperm(128)
    assert torch.equal(targets[query], point_targets(masks[query], labels, truth, counts))
    wrong_truth = truth.roll(1)
    wrong = point_targets(masks, labels, wrong_truth, counts)
    assert bool((wrong - targets).abs().max() > 0)
    union = targets.sum(-1)
    quality = targets[:, 0] / union.clamp_min(1)
    native = (literal & truth).sum(-1).double() / (literal | truth).sum(-1).clamp_min(1).double()
    assert torch.equal(quality, native)
    padded_masks = torch.cat((masks, torch.ones(128, 3, dtype=torch.bool)), dim=-1)
    padded_counts = torch.cat((counts, torch.zeros(3, dtype=counts.dtype)))
    padded = point_targets(padded_masks, labels, truth, padded_counts)
    assert torch.equal(targets, padded)
    cases.append(dict(seed=seed, points=labels.numel(), queries=128,
                      literal_count_error=float((targets - expected).abs().max()),
                      query_and_point_permutation_error=0,
                      padding_poison_error=0, oracle_quality_error=0,
                      wrong_expression_target_gap=float((wrong - targets).abs().max()),
                      empty_mask_confusion=targets[0].tolist(),
                      full_mask_confusion=targets[1].tolist()))
train = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "loss")
native_fields = [a.arg for a in train.args.args]
assert "gt_pmasks" in native_fields and "superpoints" in native_fields and "scenes_len" in native_fields
occupation = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits"]).decode().strip()
apps = subprocess.check_output([
    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits"]).decode().strip()
result = dict(
    complete=True, id="AdaRES-Y45", case_count=32, cases=cases,
    actual_parent_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    native_train_fields_present=True, native_point_masses_executed=True,
    point_target_max_error=0, minimum_wrong_expression_gap=min(c["wrong_expression_target_gap"] for c in cases),
    gpu_occupation_csv=occupation, compute_processes_csv=apps,
    scientific_status="CPU_target_mapping_verified_native_capture_pending",
    limitations=[
        "Actual parent point_masses body, but generated CPU mapping fixtures; not a real TRAIN batch.",
        "Native mask/GT expression alignment and useful confidence information remain untested.",
        "No gate PASS, no new model, no loss weight/threshold/temperature/validation tuning.",
        "Oracle quantities are TRAIN targets only; never used at inference.",
    ])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
