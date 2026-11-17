"""CPU-test native instrumentation; real native loss requires an idle GPU."""

import ast
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
from torch import nn

torch.set_num_threads(4)


def component_inputs(graph, masks, masses, tokens, collapse=False):
    """Retain every component, including background; no component filtering."""
    rows, weights, groups = [], [], []
    for query, mask in enumerate(masks):
        for side, selected in enumerate((mask, ~mask)):
            indices = np.flatnonzero(selected)
            if not len(indices):
                continue
            if collapse:
                count, labels = 1, np.zeros(len(indices), dtype=np.int32)
            else:
                count, labels = connected_components(graph[indices][:, indices], directed=False)
            component_mass = np.bincount(labels, weights=masses[indices], minlength=count)
            feature_sum = np.zeros((count, tokens.shape[1]), dtype=np.float64)
            np.add.at(feature_sum, labels, tokens[indices] * masses[indices, None])
            total_mass = masses[indices].sum()
            assert component_mass.sum() == total_mass
            assert (component_mass > 0).all()
            fraction = component_mass / total_mass
            rows.append(np.concatenate((feature_sum / component_mass[:, None], fraction[:, None]), 1))
            weights.append(fraction)
            groups.append(np.full(count, 2 * query + side, dtype=np.int64))
    return (torch.from_numpy(np.concatenate(rows)), torch.from_numpy(np.concatenate(weights)),
            torch.from_numpy(np.concatenate(groups)))


class AllComponentReadout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.component_encoder = nn.Sequential(
            nn.Linear(width + 1, width), nn.GELU(), nn.Linear(width, width))
        self.head = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, query, inputs):
        rows, weights, groups = inputs
        encoded = self.component_encoder(rows)
        pooled = encoded.new_zeros((2 * len(query), query.shape[-1]))
        pooled.index_add_(0, groups, weights[:, None] * encoded)
        pooled = pooled.reshape(len(query), -1)
        normalized_query = nn.functional.layer_norm(query, (query.shape[-1],))
        return self.head(torch.cat((normalized_query, pooled), -1)).flatten()


class FinalIndicatorAttachment:
    def __init__(self, decoder, prototype):
        self.decoder, self.prototype = decoder, prototype
        self.enabled = False
        self.wrong_correspondence = False
        self.capture, self.cached_inputs, self.residuals = {}, [], []
        self.handles = [
            decoder.register_forward_pre_hook(self.before),
            decoder.ila_ffn_layers[-1].register_forward_hook(self.tokens),
            decoder.indi_embedding.register_forward_hook(self.indicator)]

    def before(self, module, inputs):
        self.capture = dict(coordinates=inputs[5], offsets=inputs[2], calls=0)
        self.cached_inputs, self.residuals = [], []

    def tokens(self, module, inputs, output):
        self.capture["tokens"] = output

    def indicator(self, module, inputs, output):
        self.capture["calls"] += 1
        if not self.enabled or self.capture["calls"] != self.decoder.num_layer + 1:
            return output
        query, masks, padding, mass = inputs
        assert "tokens" in self.capture and output.shape[-1] == 2
        residuals = []
        for b in range(len(query)):
            valid = ~padding[b]
            start, end = [int(self.capture["offsets"][b + i]) for i in (0, 1)]
            points = self.capture["coordinates"][start:end]
            assert len(points) == int(valid.sum())
            with torch.no_grad():
                distance = torch.cdist(points, points)
                distance.fill_diagonal_(float("inf"))
                if len(points) < 2:
                    graph = np.zeros((len(points), len(points)), dtype=bool)
                else:
                    graph = (distance == distance.min(-1, keepdim=True).values).cpu().numpy()
            binary = (masks[b, :, valid] > 0).detach().cpu().numpy()
            if self.wrong_correspondence:
                binary = np.roll(binary, 1, axis=0)
            values = component_inputs(
                csr_matrix(graph), binary, mass[b, valid].detach().cpu().numpy().astype(np.float64),
                self.capture["tokens"][b, valid].detach().cpu().numpy().astype(np.float64))
            converted = tuple(value.to(query.device, dtype=query.dtype if i < 2 else torch.long)
                              for i, value in enumerate(values))
            self.cached_inputs.append(converted)
            residual = self.prototype(query[b], converted)
            self.residuals.append(residual)
            residuals.append(torch.stack((torch.zeros_like(residual), residual), -1))
        return output + torch.stack(residuals)

    def close(self):
        for handle in self.handles:
            handle.remove()


source = Path("ipdn/model_paper5/adares_w2_pointmass_competition/dec.py")
cls = next(node for node in ast.parse(source.read_text()).body
           if isinstance(node, ast.ClassDef) and node.name == "QueryCompetitionIndicator")
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
def native_mode():
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
    model = resolve_model_class(config)(**cfg.model).cuda()
    checkpoint = torch.load(
        "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/"
        "adares_w2_pointmass_competition_res/20260926_145041/best.pth",
        map_location="cpu")
    loaded = model.load_state_dict(checkpoint["model"], strict=False)
    assert not loaded.missing_keys
    del checkpoint
    model.train()
    prototype = AllComponentReadout(256).cuda()
    model.dec.add_module("y46_diagnostic_component", prototype)
    attachment = FinalIndicatorAttachment(model.dec, prototype)
    parameters_before = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in prototype.parameters())
    initial_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    rows = []

    def paired_loss(batch, enabled):
        model.zero_grad(set_to_none=True)
        for name, value in model.named_buffers():
            value.copy_(initial_buffers[name].to(value.device))
        gorilla.set_random_seed(1999)
        attachment.enabled = enabled
        attachment.wrong_correspondence = False
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        loss, _ = model(batch, mode="loss")
        assert torch.isfinite(loss)
        loss.backward()
        torch.cuda.synchronize()
        measured = dict(native_loss=float(loss), elapsed_seconds=time.monotonic() - started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        indicator_calls=attachment.capture["calls"])
        del loss
        return measured

    try:
        for index in indices:
            gorilla.set_random_seed(1999)
            batch = dataset.collate_fn([dataset[index]])
            parent = paired_loss(batch, False)
            changed = paired_loss(batch, True)
            norms = {name: float(value.grad.norm()) for name, value in prototype.named_parameters()}
            assert len(norms) == 8 and all(np.isfinite(v) and v > 0 for v in norms.values())
            assert attachment.capture["calls"] == model.dec.num_layer + 1
            rows.append(dict(dataset_index=index, scan_id=batch["scan_ids"][0],
                             parent=parent, candidate=changed, all8_native_gradient_norms=norms))
            del batch
            model.zero_grad(set_to_none=True)
            attachment.cached_inputs, attachment.residuals = [], []
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        attachment.close()
    return dict(
        complete=True, id="AdaRES-Y46", rows=rows, native_scenes=8, native_backward_executed=True,
        parent_parameters=parameters_before, added_parameters=sum(p.numel() for p in prototype.parameters()),
        minimum_native_gradient=min(v for r in rows for v in r["all8_native_gradient_norms"].values()),
        scientific_status="native_path_and_paired_cost_measurement_not_admission",
        limitations=["Real native model/loss backward, but no tools.train optimizer step or independent production package.",
                     "Each pair restores model buffers and RNG; this does not certify a bitwise full loader trajectory.",
                     "Matched disabled/candidate measurements are descriptive; no new resource threshold is fitted.",
                     "No model/checkpoint/config/queue or training state is persisted.",
                     "Production released initialization, full matched controls and resource/gate review remain required."])


if P["physical_gpu"] is not None:
    Path(P["output"]).write_text(json.dumps(native_mode()), encoding="utf-8")
    raise SystemExit(0)


class FakeDecoder(nn.Module):
    """Explicit surrogate: exercises hook ordering, not native IPDN or its loss."""

    def __init__(self):
        super().__init__()
        self.num_layer = 6
        self.ila_ffn_layers = nn.ModuleList([nn.Identity()])
        self.indi_embedding = namespace["QueryCompetitionIndicator"](256).double()
        self.query = torch.randn(2, 128, 256, dtype=torch.double)
        self.features = torch.randn(2, 16, 256, dtype=torch.double)
        self.masks = torch.randn(2, 128, 16, dtype=torch.double)
        self.padding = torch.zeros(2, 16, dtype=torch.bool)
        self.mass = torch.randint(1, 16, (2, 16)).double()

    def forward(self, x, fps_seed_sp, offsets, lang_feats, lang_masks, positions):
        self.ila_ffn_layers[-1](self.features)
        return [self.indi_embedding(self.query, self.masks, self.padding, self.mass)
                for _ in range(self.num_layer + 1)]


rows = []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    decoder = FakeDecoder()
    prototype = AllComponentReadout(256).double()
    attachment = FinalIndicatorAttachment(decoder, prototype)
    points = torch.tensor([[i % 4, i // 4, 0] for i in range(16)], dtype=torch.double).repeat(2, 1)
    inputs = (None, None, torch.tensor([0, 16, 32]), None, None, points)
    parent = decoder(*inputs)
    attachment.enabled = True
    changed = decoder(*inputs)
    assert attachment.capture["calls"] == 7 and len(attachment.cached_inputs) == 2
    assert all(torch.equal(a, b) for a, b in zip(parent[:-1], changed[:-1]))
    residual = torch.stack(attachment.residuals)
    assert torch.equal(changed[-1], parent[-1] + torch.stack((torch.zeros_like(residual), residual), -1))
    targets = torch.arange(128).remainder(2).repeat(2, 1)
    loss = nn.functional.cross_entropy(changed[-1].reshape(-1, 2), targets.flatten())
    loss.backward()
    norms = {name: float(value.grad.norm()) for name, value in prototype.named_parameters()}
    assert len(norms) == 8 and all(np.isfinite(value) and value > 0 for value in norms.values())
    with torch.no_grad():
        repeated = decoder(*inputs)
        assert all(torch.equal(a, b) for a, b in zip(changed, repeated))
        attachment.wrong_correspondence = True
        wrong = decoder(*inputs)
        attachment.enabled = False
        disabled = decoder(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(parent, disabled))
    wrong_gap = float((wrong[-1] - changed[-1]).abs().max())
    prospective = 8 * torch.finfo(torch.double).eps * 128 * max(
        1., float(changed[-1].abs().max()), float(wrong[-1].abs().max()))
    assert wrong_gap > prospective
    rows.append(dict(seed=seed, expected_indicator_calls=7, only_final_changed=True,
                     disable_parent_exact=True, exact_cached_repeat=True,
                     new_gradient_norms=norms, wrong_correspondence_gap=wrong_gap,
                     prospective8eps_queries_bound=prospective))
    attachment.close()
result = dict(
    complete=True, id="AdaRES-Y46", rows=rows, CPU_cases=32, new_tensors=8,
    minimum_synthetic_CE_gradient=min(v for r in rows for v in r["new_gradient_norms"].values()),
    minimum_wrong_correspondence_gap=min(r["wrong_correspondence_gap"] for r in rows),
    native_backward_executed=False, scientific_status="CPU_instrumentation_test_only",
    actual_W2_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    limitations=["FakeDecoder/synthetic CE explicitly do not establish native model or loss reachability.",
                 "Instrumentation changed only final foreground confidence, leaves six auxiliary outputs exact.",
                 "Same frozen all-component class/function; no graph/feature/threshold/weight search.",
                 "Full native loss, optimizer step, cost and production independence remain pending.",
                 "All existing GPU jobs preserved; no GPU allocation or new training."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
