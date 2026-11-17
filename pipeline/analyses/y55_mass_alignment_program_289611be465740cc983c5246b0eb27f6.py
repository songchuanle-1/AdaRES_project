PAYLOAD = {'id': 'AdaRES-Y55', 'token': '289611be465740cc983c5246b0eb27f6', 'sources': [{'path': 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py', 'sha256': '55160dfb062353054f6d2d5a09d971a6dc779be2e8fe34e6906187576a623ad9', 'remote_path': 'ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py', 'lf_sha256': '63c8f30bda20300d98d3d9a8a6478fc1a841cbaa3b334ee71688711e06375d97'}, {'path': 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/dec.py', 'sha256': 'd27757a6ab963c7af9476210d0b4d63aead241c33bfc28e5ebff74ef876c0b13', 'remote_path': 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py', 'lf_sha256': '7e60c1cecfc8fa7809f7c836fa6fef1366806bd23c08e2890508ea26a711acac'}, {'path': 'AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml', 'sha256': '1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03', 'remote_path': 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml', 'lf_sha256': '1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03'}], 'remote_root': '/bak_home/songcl/Project/AdaRES_project/AdaRES', 'capture_path': '/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json', 'fixtures': [{'name': 'one_scene_three_expressions', 'masses': [1, 4, 2, 7], 'offsets': [0, 4], 'scenes_len': [3]}, {'name': 'two_scenes_unequal_expression_counts', 'masses': [1, 5, 2, 4, 9, 3, 8], 'offsets': [0, 3, 7], 'scenes_len': [1, 3]}, {'name': 'three_scenes_nonuniform_support', 'masses': [2, 7, 1, 6, 3, 9, 4, 8, 5], 'offsets': [0, 2, 5, 9], 'scenes_len': [3, 1, 2]}]}
"""CPU-only contract test; PAYLOAD is frozen by the local lease holder."""
import ast
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import gorilla
import torch
import torch.nn as nn

assert socket.gethostname() == "dl01-MS-7C60"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
torch.set_num_threads(2)
torch.manual_seed(1999)
ROOT = Path(PAYLOAD["remote_root"])
os.chdir(str(ROOT))
for binding in PAYLOAD["sources"]:
    raw = (ROOT / binding["remote_path"]).read_bytes()
    assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == binding["lf_sha256"]


class CpuFpsFixture:
    """FPS is irrelevant to expansion order and is not measured here."""

    calls = 0

    @classmethod
    def apply(cls, positions, count):
        cls.calls += 1
        return (torch.arange(count) % positions.shape[1]).unsqueeze(0).int()


def selected_class(path, original, replacement, methods=None, bases=None):
    tree = ast.parse(path.read_text())
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == original)
    result = copy.deepcopy(source)
    result.name = replacement
    if methods is not None:
        result.body = [node for node in result.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        assert {node.name for node in result.body} == set(methods)
    if bases is not None:
        result.bases = bases
    module = ast.fix_missing_locations(ast.Module(body=[result], type_ignores=[]))
    scope = dict(torch=torch, nn=nn, FurthestPointSampling=CpuFpsFixture)
    exec(compile(module, str(path), "exec"), scope)
    return scope[replacement]


parent = ROOT / "ipdn/model_paper5/adares_w2_pointmass_competition"
Bookkeeping = selected_class(parent / "ipdn.py", "IPDN", "Bookkeeping",
    methods=["point_masses", "expand_and_fps"], bases=[])
Packing = selected_class(parent / "dec.py", "DEC", "Packing",
    methods=["get_batches"], bases=[])
Indicator = selected_class(parent / "dec.py", "QueryCompetitionIndicator", "Indicator")
bookkeeping = Bookkeeping()
bookkeeping.fps_num = 256
packing = Packing()
indicator = Indicator(256)
indicator.eval()


def run_case(name, superpoints, offsets, scenes_len, source_kind, metadata=None):
    superpoints = torch.as_tensor(superpoints, dtype=torch.long)
    offsets = torch.as_tensor(offsets, dtype=torch.long)
    scenes_len = [int(value) for value in scenes_len]
    assert len(scenes_len) == len(offsets) - 1 and min(scenes_len) > 0
    total = int(offsets[-1])
    identity = torch.arange(total, dtype=torch.float32).unsqueeze(-1)
    coords = torch.cat((identity, identity * 2, identity * 3), dim=1)
    image = torch.cat((identity, identity + 0.25), dim=1)
    labels = [torch.arange(int(offsets[i + 1] - offsets[i])) for i in range(len(scenes_len))]
    masses = bookkeeping.point_masses(superpoints, offsets, scenes_len)
    expanded = bookkeeping.expand_and_fps(identity, coords, offsets, labels, scenes_len, image, mode="tr")
    features, positions, seeds, expanded_offsets, expanded_labels, images = expanded
    counter = Counter(int(value) for value in superpoints.tolist())
    expected = torch.tensor([counter[int(value)] for value in features[:, 0]], dtype=torch.float32)
    owners = []
    expected_offsets = [0]
    expected_ids = []
    totals = []
    for scene, copies in enumerate(scenes_len):
        ids = list(range(int(offsets[scene]), int(offsets[scene + 1])))
        for _ in range(copies):
            owners.append(scene)
            expected_ids.extend(ids)
            expected_offsets.append(expected_offsets[-1] + len(ids))
            totals.append(sum(counter[value] for value in ids))
    mismatches = int((masses != expected).sum())
    assert features[:, 0].long().tolist() == expected_ids
    assert expanded_offsets.tolist() == expected_offsets
    assert seeds.shape == (sum(scenes_len), 256)
    assert len(expanded_labels) == sum(scenes_len)
    assert torch.equal(positions[:, 0], features[:, 0])
    assert torch.equal(images[:, 0], features[:, 0])
    packed, padding = packing.get_batches(masses.unsqueeze(-1), expanded_offsets)
    reference, reference_padding = packing.get_batches(expected.unsqueeze(-1), expanded_offsets)
    packed = packed.squeeze(-1)
    reference = reference.squeeze(-1)
    assert torch.equal(padding, reference_padding)
    assert torch.equal(packed.sum(-1), torch.tensor(totals, dtype=torch.float32))
    assert torch.equal(packed.masked_select(padding), torch.zeros(int(padding.sum())))
    assert not bool(padding.all(-1).any())
    generator = torch.Generator().manual_seed(1999)
    queries = torch.randn(sum(scenes_len), 128, 256, generator=generator)
    logits = torch.randn(sum(scenes_len), 128, packed.shape[1], generator=generator)
    wrong = packed.clone()
    changed_positions = 0
    for row in range(len(owners)):
        valid = int((~padding[row]).sum())
        wrong[row, :valid] = packed[row, :valid].roll(1)
        changed_positions += int((wrong[row] != packed[row]).sum())
    with torch.no_grad():
        native = indicator(queries, logits, padding, packed)
        repeated = indicator(queries, logits, padding, packed)
        independent = indicator(queries, logits, padding, reference)
        shifted = indicator(queries, logits, padding, wrong)
    repeat_delta = float((native - repeated).abs().max())
    reference_delta = float((native - independent).abs().max())
    shifted_delta = float((native - shifted).abs().max())
    assert bool(torch.isfinite(native).all())
    assert repeat_delta == 0 and reference_delta == 0
    assert mismatches == 0
    assert changed_positions > 0 and shifted_delta > 0
    return dict(name=name, source_kind=source_kind, metadata=metadata,
        scenes=len(scenes_len), expressions=sum(scenes_len), scenes_len=scenes_len,
        superpoint_counts=(offsets[1:] - offsets[:-1]).tolist(),
        original_points=int(superpoints.numel()), expanded_mass_entries=int(masses.numel()),
        mass_mapping_mismatches=mismatches, packed_padding_entries=int(padding.sum()),
        expansion_ids_exact=True, offsets_exact=True, point_totals_exact=True,
        packed_padding_mass_zero=True, native_indicator_reference_max_abs=reference_delta,
        native_indicator_repeat_max_abs=repeat_delta,
        deliberately_shifted_mass_positions=changed_positions,
        deliberately_shifted_mass_indicator_max_abs=shifted_delta)


started = time.perf_counter()
rows = []
for fixture in PAYLOAD["fixtures"]:
    masses = fixture["masses"]
    ids = [index for index, count in enumerate(masses) for _ in range(count)]
    rows.append(run_case(fixture["name"], ids, fixture["offsets"], fixture["scenes_len"],
        "deterministic_bookkeeping_fixture_not_native_FPS"))

from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

capture_path = Path(PAYLOAD["capture_path"])
capture_raw = capture_path.read_bytes()
capture = json.loads(capture_raw)
frozen = capture["rows"]
assert len(frozen) == 8
config = ROOT / "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
cfg = gorilla.Config.fromfile(str(config))
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
for pair in range(0, 8, 2):
    items = []
    metadata = []
    for scene in frozen[pair:pair + 2]:
        gorilla.set_random_seed(1999)
        item = dataset[scene["dataset_index"]]
        single = dataset.collate_fn([copy.deepcopy(item)])
        assert single["scan_ids"][0] == scene["scan_id"]
        assert [str(value) for value in single["ann_ids"]] == [value["ann_id"] for value in scene["expressions"]]
        items.append(item)
        metadata.append(dict(dataset_index=scene["dataset_index"], scan_id=scene["scan_id"],
            ann_ids=[str(value) for value in single["ann_ids"]]))
    batch = dataset.collate_fn(copy.deepcopy(items))
    assert all(not torch.is_tensor(value) or value.device.type == "cpu" for value in batch.values())
    rows.append(run_case("frozen_TRAIN_pair_" + str(pair // 2), batch["superpoints"],
        batch["batch_offsets"], batch["scenes_len"], "actual_frozen_TRAIN_collate_CPU", metadata))

assert not torch.cuda.is_initialized()
result = dict(complete=True, id=PAYLOAD["id"], token=PAYLOAD["token"],
    hostname=socket.gethostname(), cpu_only=True, cuda_initialized=False,
    sources_verified=True, capture_sha256=hashlib.sha256(capture_raw).hexdigest(),
    actual_TRAIN_scenes=8, actual_TRAIN_expressions=32,
    fixture_cases=len(PAYLOAD["fixtures"]), actual_TRAIN_pairs=4,
    original_IPDN_methods_AST_unchanged=True, original_DEC_packing_AST_unchanged=True,
    original_indicator_class_AST_unchanged=True,
    FPS_substitution="Deterministic CPU index fixture; no actual FPS, sparse forward, checkpoint or model training.",
    mass_mapping_mismatches=sum(row["mass_mapping_mismatches"] for row in rows),
    all_positive_controls_active=all(row["deliberately_shifted_mass_indicator_max_abs"] > 0 for row in rows),
    indicator_reference_max_abs=max(row["native_indicator_reference_max_abs"] for row in rows),
    indicator_repeat_max_abs=max(row["native_indicator_repeat_max_abs"] for row in rows),
    elapsed_seconds=time.perf_counter() - started, rows=rows,
    new_model_files=0, optimizer_steps=0, admissions=0, training_launches=0,
    limitation="Tests data-expansion bookkeeping, not learned efficacy, whole-model native/backward/resource gates or global absence of bugs.")
print("Y55_RESULT=" + json.dumps(result, separators=(",", ":")))
