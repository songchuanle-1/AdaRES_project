"""Inherited nearest-tie graph witnesses and prediction-blind native capture."""

import ast
import base64
import gc
import hashlib
import json
from pathlib import Path
import time
import zlib
import gorilla
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
from torch import nn
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class

assert P["physical_gpu"] == 1
torch.set_num_threads(4)
source = Path("ipdn/model_paper5/adares_y36_local_adjacency/dec.py")
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MaskLocalAdjacencyReadout")
assert any(isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == "distance"
           for n in ast.walk(cls))
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)


def adjacency(points):
    if len(points) < 2:
        return np.zeros((len(points), len(points)), dtype=bool)
    distance = torch.cdist(points, points)
    distance.fill_diagonal_(float("inf"))
    closest = distance.min(-1, keepdim=True).values
    return (distance == closest).detach().cpu().numpy()


def components(graph, mask, masses):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    sub = csr_matrix(graph)[mask][:, mask]
    count, labels = connected_components(sub, directed=False)
    result = np.bincount(labels, weights=np.asarray(masses)[mask], minlength=count)
    return sorted(result.tolist())


def pack(value):
    value = np.ascontiguousarray(value)
    raw = value.tobytes()
    return dict(dtype=value.dtype.str, shape=list(value.shape),
                sha256=hashlib.sha256(raw).hexdigest(),
                zlib_base64=base64.b64encode(zlib.compress(raw)).decode())


cases = []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    readout = namespace["MaskLocalAdjacencyReadout"](256).double()
    left = torch.tensor([[0,0,0],[1,0,0],[2,0,0],[2,1,0],
                         [2,2,0],[1,2,0],[0,2,0],[0,1,0]], dtype=torch.double)
    right = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                          [4,0,0],[5,0,0],[5,1,0],[4,1,0]], dtype=torch.double)
    query = torch.randn(1, 128, 256, dtype=torch.double)
    tokens = torch.randn(1, 1, 256, dtype=torch.double).expand(1, 8, 256).clone()
    masks = torch.full((1, 128, 8), 8., dtype=torch.double)
    pad = torch.zeros(1, 8, dtype=torch.bool)
    mass = torch.ones(1, 8, dtype=torch.double)
    with torch.no_grad():
        a = readout(query, masks, pad, mass, tokens, left.unsqueeze(0))
        b = readout(query, masks, pad, mass, tokens, right.unsqueeze(0))
    ga, gb = adjacency(left), adjacency(right)
    assert np.array_equal(ga.sum(1), gb.sum(1))
    assert (ga.sum(1) == 2).all()
    ma = components(ga, np.ones(8, bool), np.ones(8))
    mb = components(gb, np.ones(8, bool), np.ones(8))
    assert ma == [8.] and mb == [4., 4.]
    gap = float((a - b).abs().max())
    bound = 8 * torch.finfo(torch.double).eps * max(1., float(a.abs().max()), float(b.abs().max()))
    assert gap <= bound
    order = torch.randperm(8).numpy()
    assert ma == components(ga[order][:, order], np.ones(8, bool), np.ones(8)[order])
    cases.append(dict(seed=seed, actual_Y36_readout_difference=gap,
                      prospective8eps_bound=bound, component_masses=[ma, mb],
                      local_degree_equal=True, permutation_mass_error=0))
gorilla.set_random_seed(1999)
config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
cfg = gorilla.Config.fromfile(config)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
indices = [0, 44, 60, 101, 115, 126, 137, 171]
assert len({dataset.scene_inputs[i][0]["scene_id"] for i in indices}) == 8
model = resolve_model_class(config)(**cfg.model).cuda()
checkpoint = torch.load("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth", map_location="cpu")
loaded = model.load_state_dict(checkpoint["model"], strict=False)
assert not loaded.missing_keys
del checkpoint
model.train()
capture = {}


def dec_pre(module, inputs):
    capture["coordinates"] = inputs[5].detach().clone()
    capture["offsets"] = inputs[2].detach().clone()


def indi_pre(module, inputs):
    capture["q"], capture["masks"], capture["padding"], capture["mass"] = [
        v.detach().clone() for v in inputs]


hooks = [
    model.dec.register_forward_pre_hook(dec_pre),
    model.dec.indi_embedding.register_forward_pre_hook(indi_pre),
    model.dec.ila_ffn_layers[-1].register_forward_hook(
        lambda module, inputs, output: capture.update(tokens=output.detach().clone()))]
rows = []
for index in indices:
    gorilla.set_random_seed(1999)
    batch = dataset.collate_fn([dataset[index]])
    capture.clear()
    with torch.no_grad():
        loss, _ = model(batch, mode="loss")
    assert torch.isfinite(loss)
    expressions = []
    for b in range(capture["q"].shape[0]):
        start, end = int(capture["offsets"][b]), int(capture["offsets"][b + 1])
        points = capture["coordinates"][start:end]
        valid = ~capture["padding"][b]
        assert end - start == int(valid.sum())
        graph = adjacency(points)
        masses = capture["mass"][b, valid].cpu().numpy()
        masks = capture["masks"][b, :, valid].cpu().numpy()
        assert masks.shape == (128, len(points))
        binary = masks > 0
        gt = batch["gt_spmasks"][b].detach().cpu().reshape(-1).numpy() > .5
        assert len(gt) == len(points)
        per_query = [components(graph, mask, masses) for mask in binary]
        for mask, values in zip(binary, per_query):
            assert sum(values) == float(masses[mask].sum())
        order = np.arange(len(points) - 1, -1, -1)
        checks = [components(graph[order][:, order], mask[order], masses[order]) for mask in binary]
        assert checks == per_query
        gt_masses = components(graph, gt, masses)
        graph_edges = np.stack(np.nonzero(graph), -1).astype(np.int32)
        packed = dict(query=pack(capture["q"][b].cpu().numpy()),
                      tokens=pack(capture["tokens"][b, valid].cpu().numpy()),
                      masks=pack(masks), masses=pack(masses),
                      coordinates=pack(points.cpu().numpy()), graph_edges=pack(graph_edges))
        expressions.append(dict(
            ann_id=str(batch["ann_ids"][b]), object_ids=batch["object_idss"][b],
            selected_mass_error=0, component_mass_permutation_error=0,
            foreground_component_masses=per_query, GT_superpoint_component_masses=gt_masses,
            graph_nodes=len(points), graph_edges=len(graph_edges),
            graph_support_density=float(graph.mean()), packed_native_inputs=packed))
    rows.append(dict(dataset_index=index, scan_id=batch["scan_ids"][0], expressions=expressions))
    del batch, loss
    capture.clear()
    gc.collect()
    torch.cuda.empty_cache()
for hook in hooks:
    hook.remove()
result = dict(
    complete=True, id="AdaRES-Y46", cases=cases, rows=rows, CPU_cases=32,
    native_scenes=8, native_expressions=sum(len(r["expressions"]) for r in rows),
    native_queries=128, inherited_graph="Exact Y36 torch.cdist/min/nearest-distance ties; no k, radius or selected threshold.",
    component_engine="scipy.sparse.csgraph.connected_components(directed=False), inherited weak nearest-edge support",
    scipy_version=scipy.__version__, actual_Y36_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    actual_Y36_class_source=ast.get_source_segment(source.read_text(), cls),
    maximum_CPU_parent_difference=max(c["actual_Y36_readout_difference"] for c in cases),
    multiple_component_queries=sum(len(v) > 1 for r in rows for e in r["expressions"] for v in e["foreground_component_masses"]),
    GT_multiple_component_expressions=sum(len(e["GT_superpoint_component_masses"]) > 1 for r in rows for e in r["expressions"]),
    scientific_status="CPU_graph_information_witness_and_native_incidence_only",
    limitations=["Sparse graph witnesses use real Y36 readout, not trained efficacy or all possible parent summaries.",
                 "Actual TRAIN masks/geometry and full128 query component incidence; no query/scene selection from outcomes.",
                 "True referents can have multiple inherited-graph components: no largest-component filtering or one-component prior.",
                 "Native full Y46 readout/backward/resources and learned usefulness remain untested.",
                 "Graph relabeling checks component masses; no global CUDA geometry/kernel bitwise claim.",
                 "No source/config/checkpoint updated, no oracle inference or new training."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
