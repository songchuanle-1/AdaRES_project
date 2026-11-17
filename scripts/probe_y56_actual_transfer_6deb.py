"""Append to native W2 setup: fixed-focal transfer in discovered duplicate cases."""

capture_path = Path("/tmp/adares_y56_broader_unique_6debf88fed634a13b772f27bdc244bbd.json")
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == (
    "c02ac3d48780e9b5557e9edacaecdcc8a98b9c27d0b35074fb4ee7e14a607cee")
capture = json.loads(capture_path.read_text())
pairs = sorted({r["pair"] for r in capture["rows"] if r["joint_duplicate_excess"]})
assert pairs == [8]
cases = []
started = time.perf_counter()


def corrected(module, query, masks, mass, inverse, counts):
    probabilities = masks.sigmoid()
    weighted = probabilities * mass.unsqueeze(0)
    areas = weighted.sum(-1)
    intersection = weighted @ probabilities.T
    overlap = intersection / (
        areas[:, None] + areas[None, :] - intersection).clamp_min(1e-6)
    overlap.fill_diagonal_(0)
    weights = overlap / counts[inverse].to(query.dtype).unsqueeze(0)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    normalized = module.query_norm(query)
    context = weights @ normalized
    evidence = torch.cat((
        normalized, context, normalized - context,
        overlap.max(-1).values.unsqueeze(-1)), -1)
    return module.head(module.evidence_norm(evidence))


for pair in pairs:
    items, identities = [], []
    for scene in capture["cohort"][pair * 2:pair * 2 + 2]:
        gorilla.set_random_seed(1999)
        item = dataset[scene["dataset_index"]]
        single = dataset.collate_fn([copy.deepcopy(item)])
        assert str(single["scan_ids"][0]) == scene["scene"]
        assert [str(v) for v in single["ann_ids"]] == scene["ann_ids"]
        items.append(item)
        identities.extend((scene["scene"], ann) for ann in scene["ann_ids"])
    batch = dataset.collate_fn(copy.deepcopy(items))
    targets = []
    expression = 0
    for scene_index, length in enumerate(batch["scenes_len"]):
        start, end = [int(v) for v in batch["batch_offsets"][scene_index:scene_index + 2]]
        selected = (batch["superpoints"] >= start) & (batch["superpoints"] < end)
        membership = batch["superpoints"][selected].long() - start
        total = torch.bincount(membership, minlength=end - start)
        for _ in range(int(length)):
            gt = batch["gt_pmasks"][expression].reshape(-1)
            assert gt.numel() == membership.numel()
            positive = torch.bincount(
                membership, weights=(gt > 0.5).float(), minlength=len(total))
            targets.append((total, positive, hashlib.sha256(
                gt.contiguous().numpy().tobytes()).hexdigest()))
            expression += 1
    calls = []

    def observe(module, inputs, output):
        layer = len(calls)
        calls.append(layer)
        for index, (scene, ann) in enumerate(identities):
            q = inputs[0][index]
            valid = ~inputs[2][index]
            m = inputs[1][index, :, valid]
            mass = inputs[3][index, valid]
            unique, inverse, counts = torch.unique(
                torch.cat((q, m), dim=1), dim=0, return_inverse=True, return_counts=True)
            row = dict(scene=scene, ann_id=ann, layer=layer, focal=0,
                       duplicate_excess=int(128 - len(unique)))
            if len(unique) == 128:
                row["transfer"] = "not_applicable_no_exact_joint_copy"
                cases.append(row)
                continue
            if int(counts[inverse[0]]) != 1:
                row["transfer"] = "not_applicable_preregistered_focal_not_unique"
                cases.append(row)
                continue
            removable = [i for i in range(1, 128) if int(counts[inverse[i]]) > 1]
            assert removable
            removed = removable[-1]
            recipients = [i for i in range(1, 128) if inverse[i] != inverse[removed]]
            assert recipients
            recipient = recipients[0]
            changed_q, changed_m = q.clone(), m.clone()
            changed_q[removed], changed_m[removed] = q[recipient], m[recipient]
            changed_unique, changed_inverse, changed_counts = torch.unique(
                torch.cat((changed_q, changed_m), 1), dim=0,
                return_inverse=True, return_counts=True)
            assert torch.equal(unique, changed_unique)
            assert torch.equal(q[0], changed_q[0]) and torch.equal(m[0], changed_m[0])
            padding = torch.zeros((1, m.shape[-1]), dtype=torch.bool, device=q.device)

            def native(query, masks):
                return module.forward(query.unsqueeze(0), masks.unsqueeze(0),
                                      padding, mass.unsqueeze(0))[0]

            original = native(q, m)
            repeated = native(q, m)
            transferred = native(changed_q, changed_m)
            perm = torch.roll(torch.arange(128, device=q.device), 1)
            permuted = native(q[perm], m[perm])[torch.argsort(perm)]
            dedup = corrected(module, q, m, mass, inverse, counts)
            dedup_transferred = corrected(
                module, changed_q, changed_m, mass, changed_inverse, changed_counts)
            total, positive, gt_hash = targets[index]
            total, positive = total.to(q.device), positive.to(q.device)
            assert torch.equal(total.to(mass.dtype), mass)
            binary = (m[0].sigmoid() >= 0.5).to(q.dtype)
            numerator = (binary * positive).sum()
            quality = numerator / (
                (binary * total).sum() + positive.sum() - numerator + 1e-6)
            bound = 128 * torch.finfo(q.dtype).eps * max(
                1.0, float(original.abs().max()), float(dedup.abs().max()))
            row.update(
                transfer="executed", removed_copy_index=removed,
                recipient_prototype_index=recipient,
                unique_set_and_focal_unchanged=True,
                focal_point_quality_unchanged=float(quality),
                actual_point_GT_sha256=gt_hash,
                native_repeat_max_abs=float((original - repeated).abs().max()),
                native_permutation_max_abs=float((original - permuted).abs().max()),
                native_focal_logit_change=float((original[0] - transferred[0]).abs().max()),
                native_focal_confidence_change=float(abs(
                    original[0].softmax(-1)[1] - transferred[0].softmax(-1)[1])),
                inverse_multiplicity_focal_change=float(
                    (dedup[0] - dedup_transferred[0]).abs().max()),
                arithmetic_bound=bound)
            assert row["inverse_multiplicity_focal_change"] <= bound
            cases.append(row)

    handle = model.dec.indi_embedding.register_forward_hook(observe)
    try:
        gorilla.set_random_seed(1999)
        with torch.no_grad():
            loss, _ = model(copy.deepcopy(batch), mode="loss")
        assert torch.isfinite(loss).all() and len(calls) == 7
    finally:
        handle.remove()
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    del loss, batch, items
assert parameter_versions == {name: value._version for name, value in model.named_parameters()}
assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
assert digest(checkpoint_path) == FROZEN["checkpoint_sha256"]
executed = [r for r in cases if r["transfer"] == "executed"]
result = dict(
    complete=True, id="AdaRES-Y56", fixed_focal=0, discovery_followup_pairs=pairs,
    native_sparse_TRAIN_batch2=True, expressions=8, expression_layer_records=len(cases),
    actual_duplicate_records=sum(r["duplicate_excess"] > 0 for r in cases),
    transfer_cases=len(executed), cases=cases,
    parameters_unchanged=True, buffers_restored=True, optimizer_steps=0,
    weights_saved=False, admission=False, elapsed_seconds=time.perf_counter() - started,
    limitations="Result-conditioned case follow-up, NOT new prevalence evidence. Same unique "
                "competitor set and unchanged focal point IoU do not establish prediction harm "
                "or trained efficacy. Inverse multiplicity is a diagnostic reference only; "
                "no candidate implementation, new layers/thresholds or training.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
