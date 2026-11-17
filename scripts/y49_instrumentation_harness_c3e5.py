"""Harness body; builder prepends the exact frozen descriptor and readout AST."""

class FinalResolventAttachment:
    def __init__(self, decoder, readout):
        self.decoder, self.readout = decoder, readout
        self.enabled = False
        self.wrong_correspondence = False
        self.cached, self.capture = [], {}
        self.handles = [decoder.register_forward_pre_hook(self.before),
                        decoder.indi_embedding.register_forward_hook(self.indicator)]

    def before(self, module, inputs):
        self.capture = dict(coordinates=inputs[5], offsets=inputs[2], calls=0)
        self.cached = []

    def indicator(self, module, inputs, output):
        self.capture["calls"] += 1
        if not self.enabled or self.capture["calls"] != self.decoder.num_layer + 1:
            return output
        query, masks, padding, masses = inputs
        residuals = []
        for b in range(len(query)):
            valid = ~padding[b]
            start, end = [int(self.capture["offsets"][b + i]) for i in (0, 1)]
            points = self.capture["coordinates"][start:end]
            assert len(points) == int(valid.sum())
            with torch.no_grad():
                distance = torch.cdist(points, points)
                distance.fill_diagonal_(float("inf"))
                graph = csr_matrix(
                    (distance == distance.min(-1, keepdim=True).values).cpu().numpy()
                    if len(points) > 1 else np.zeros((len(points), len(points)), dtype=bool))
            binary = (masks[b, :, valid] > 0).detach().cpu().numpy()
            if self.wrong_correspondence:
                binary = np.roll(binary, 1, axis=0)
            values, _, _ = descriptors(
                graph, binary, masses[b, valid].detach().cpu().numpy().astype(np.float64))
            values = values.to(query.device, dtype=query.dtype)
            residual = self.readout(query[b], values)
            self.cached.append((query[b].detach().clone(), values.detach().clone()))
            residuals.append(torch.stack((torch.zeros_like(residual), residual), -1))
        return output + torch.stack(residuals)

    def close(self):
        for handle in self.handles:
            handle.remove()


def native_mode():
    import gc
    import time
    import gorilla
    from ipdn.dataset import build_dataset
    from ipdn.utils import get_root_logger
    from train import resolve_model_class
    assert P["physical_gpu"] in (0, 1) and torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    gorilla.set_random_seed(1999)
    config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
    cfg = gorilla.Config.fromfile(config)
    dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
    indices = [0, 44, 60, 101, 115, 126, 137, 171]
    assert len({dataset.scene_inputs[index][0]["scene_id"] for index in indices}) == 8
    model = resolve_model_class(config)(**cfg.model).cuda()
    checkpoint = torch.load(
        "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth",
        map_location="cpu")
    loaded = model.load_state_dict(checkpoint["model"], strict=False)
    assert not loaded.missing_keys
    assert not set(loaded.unexpected_keys) - {
        "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
    del checkpoint
    model.train()
    readout = GlobalResolventReadout(256).cuda()
    model.dec.add_module("y49_diagnostic_resolvent", readout)
    attachment = FinalResolventAttachment(model.dec, readout)
    buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    rows = []

    def paired(batch, enabled):
        model.zero_grad(set_to_none=True)
        for name, value in model.named_buffers():
            value.copy_(buffers[name].to(value.device))
        gorilla.set_random_seed(1999)
        attachment.enabled = enabled
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        loss, _ = model(batch, mode="loss")
        assert torch.isfinite(loss)
        loss.backward()
        torch.cuda.synchronize()
        result = dict(loss=float(loss), seconds=time.monotonic() - start,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        del loss
        return result

    try:
        for index in indices:
            gorilla.set_random_seed(1999)
            batch = dataset.collate_fn([dataset[index]])
            parent = paired(batch, False)
            candidate = paired(batch, True)
            gradients = {name: float(value.grad.norm()) for name, value in readout.named_parameters()
                         if value.grad is not None}
            assert len(gradients) == 4 and all(np.isfinite(value) and value > 0
                                             for value in gradients.values())
            with torch.no_grad():
                cached = []
                for query, values in attachment.cached:
                    correct = readout(query, values)
                    assert torch.equal(correct, readout(query, values))
                    wrong = readout(query, values.roll(1, dims=0))
                    disabled_information = readout(query, torch.zeros_like(values))
                    cached.append(dict(wrong_gap=float((correct - wrong).abs().max()),
                                       information_gap=float((correct - disabled_information).abs().max()),
                                       repeat_exact=True))
            rows.append(dict(dataset_index=index, scan_id=batch["scan_ids"][0],
                             parent=parent, candidate=candidate, native_gradients=gradients,
                             cached_controls=cached))
            del batch
            attachment.cached = []
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        attachment.close()
    return dict(
        complete=True, id="AdaRES-Y49", native_scenes=8, native_backward_executed=True,
        prototype_parameters=66561, minimum_native_gradient=min(
            value for row in rows for value in row["native_gradients"].values()), rows=rows,
        scientific_status="native_diagnostic_reachability_not_production_or_admission",
        limitations=[
            "Existing W2 checkpoint plus temporary four-tensor head, not a released-init candidate training run.",
            "Descriptive paired full native loss/time/memory only; no trained efficacy or lightweight Pareto PASS.",
            "Controls use exact cached indicator inputs; full-loader bitwise repeat is not claimed.",
            "Independent production, released-init actual tools.train updates/restore, full resource/gate review remain pending.",
            "No diagnostic weights or model/config/queue state persisted."])


def cpu_mode():
    import ast
    source = Path(P["root"]) / "ipdn/model_paper5/adares_w2_pointmass_competition/dec.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "QueryCompetitionIndicator")
    namespace = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)

    class HookHarness(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_layer = 6
            self.indi_embedding = namespace["QueryCompetitionIndicator"](256).double()
            self.query = torch.randn(2, 128, 256, dtype=torch.double)
            self.masks = torch.randn(2, 128, 16, dtype=torch.double)
            self.padding = torch.zeros(2, 16, dtype=torch.bool)
            self.mass = torch.randint(1, 16, (2, 16)).double()

        def forward(self, x, fps, offsets, language, language_masks, points):
            return [self.indi_embedding(self.query, self.masks, self.padding, self.mass)
                    for _ in range(self.num_layer + 1)]

    rows = []
    for seed in range(32):
        torch.manual_seed(1999 + seed)
        decoder, readout = HookHarness(), GlobalResolventReadout(256).double()
        attachment = FinalResolventAttachment(decoder, readout)
        points = torch.tensor([[i % 4, i // 4, 0] for i in range(16)],
                              dtype=torch.double).repeat(2, 1)
        inputs = (None, None, torch.tensor([0, 16, 32]), None, None, points)
        parent = decoder(*inputs)
        attachment.enabled = True
        candidate = decoder(*inputs)
        assert all(torch.equal(left, right) for left, right in zip(parent[:-1], candidate[:-1]))
        assert torch.equal(parent[-1][..., 0], candidate[-1][..., 0])
        assert attachment.capture["calls"] == 7 and len(attachment.cached) == 2
        targets = torch.arange(128).remainder(2).repeat(2, 1)
        nn.functional.cross_entropy(candidate[-1].reshape(-1, 2), targets.flatten()).backward()
        gradients = {name: float(value.grad.norm()) for name, value in readout.named_parameters()}
        assert len(gradients) == 4 and all(np.isfinite(value) and value > 0
                                         for value in gradients.values())
        with torch.no_grad():
            repeated = decoder(*inputs)
            assert all(torch.equal(left, right) for left, right in zip(candidate, repeated))
            attachment.wrong_correspondence = True
            wrong = decoder(*inputs)
            attachment.enabled = False
            disabled = decoder(*inputs)
        assert all(torch.equal(left, right) for left, right in zip(parent, disabled))
        wrong_gap = float((candidate[-1] - wrong[-1]).abs().max())
        assert wrong_gap > 8 * torch.finfo(torch.double).eps * 128 * max(
            1.0, float(candidate[-1].abs().max()), float(wrong[-1].abs().max()))
        rows.append(dict(seed=seed, only_final_foreground_changed=True, auxiliary6_exact=True,
                         disabled_parent_exact=True, exact_repeat=True, wrong_gap=wrong_gap,
                         gradients=gradients))
        attachment.close()
    return dict(
        complete=True, id="AdaRES-Y49", CPU_cases=32, new_tensors=4,
        prototype_parameters=66561, minimum_CPU_CE_gradient=min(
            value for row in rows for value in row["gradients"].values()),
        minimum_wrong_correspondence_gap=min(row["wrong_gap"] for row in rows), rows=rows,
        native_backward_executed=False,
        scientific_status="CPU_final_indicator_instrumentation_only",
        limitations=["HookHarness CE and exact source functions are not native model/loss execution.",
                     "No GPU, independent production package, native optimizer/resource/efficacy or admission claim."])


torch.set_num_threads(4)
result = cpu_mode() if P["physical_gpu"] is None else native_mode()
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
