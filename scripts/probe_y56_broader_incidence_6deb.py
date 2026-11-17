"""Append to the frozen Y56 setup: prediction-blind disjoint TRAIN cohort."""

excluded = {row["dataset_index"] for row in FROZEN["scenes"]}
indices = [i for i in range(len(dataset)) if i not in excluded][:32]
assert len(indices) == 32 and not (set(indices) & excluded)
records, cohort = [], []
started = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
try:
    for pair in range(0, 32, 2):
        items, identities = [], []
        for index in indices[pair:pair + 2]:
            gorilla.set_random_seed(1999)
            item = dataset[index]
            single = dataset.collate_fn([copy.deepcopy(item)])
            assert len(single["scan_ids"]) == 1
            scene = str(single["scan_ids"][0])
            anns = [str(value) for value in single["ann_ids"]]
            assert len(anns) == 4
            cohort.append(dict(dataset_index=index, scene=scene, ann_ids=anns))
            identities.extend((scene, ann) for ann in anns)
            items.append(item)
        batch = dataset.collate_fn(copy.deepcopy(items))
        calls = []

        def observe(_module, inputs):
            query, masks, padding, mass = inputs
            assert query.shape == (8, 128, 256) and padding.shape == mass.shape
            layer = len(calls)
            for index, (scene, ann) in enumerate(identities):
                valid = ~padding[index]
                q = query[index].detach().cpu()
                m = masks[index, :, valid].detach().cpu()
                assert torch.isfinite(q).all() and torch.isfinite(m).all()
                stats, _, _ = incidence(q, m)
                records.append(dict(pair=pair // 2, layer=layer, scene=scene,
                                    ann_id=ann, valid_superpoints=int(valid.sum()), **stats))
            calls.append(layer)

        hook = model.dec.indi_embedding.register_forward_pre_hook(observe)
        try:
            gorilla.set_random_seed(1999)
            with torch.no_grad():
                loss, _ = model(copy.deepcopy(batch), mode="loss")
            assert torch.isfinite(loss).all() and len(calls) == 7
        finally:
            hook.remove()
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        del loss, batch, items
finally:
    for name, value in model.named_buffers():
        value.copy_(buffers[name])
assert len({row["scene"] for row in cohort}) == 32
assert not ({row["scene"] for row in cohort} & {row["scan_id"] for row in FROZEN["scenes"]})
assert len(records) == 32 * 4 * 7
assert parameter_versions == {name: value._version for name, value in model.named_parameters()}
assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
assert digest(checkpoint_path) == FROZEN["checkpoint_sha256"]
assert (root / ".codex_source_commit").read_text().strip() == FROZEN["commit"]
result = dict(
    complete=True, id="AdaRES-Y56", native_TRAIN_batch2=True, actual_sparse_FPS=True,
    cohort=cohort, fixed_scenes=32, expressions=128, indicator_calls=112,
    expression_layer_records=len(records),
    checkpoint_sha256=FROZEN["checkpoint_sha256"],
    parameters_unchanged=True, buffers_restored=True,
    exact_joint_duplicate_excess=sum(r["joint_duplicate_excess"] for r in records),
    hard_mask_duplicate_excess=sum(r["hard_mask_duplicate_excess"] for r in records),
    rows=records, elapsed_seconds=time.perf_counter() - started,
    peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    optimizer_steps=0, weights_saved=False, training_launches=0, admission=False,
    limitation="Disjoint fixed32TRAIN scenes, first32dataset indices excluding old8. "
               "No label/prediction selection. Correlated queries/layers and deterministic "
               "index cohort are not a population prevalence sample. Hardmask equality "
               "does not establish exact joint representation copying.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
