"""Load exact diagnostic package bytes inside project-owned fixture space."""

import ast
import base64
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import gorilla
import numpy as np
import torch
from torch import nn
import yaml
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

root = Path(P["root"])
assert (root / ".codex_source_commit").read_text().strip() == PAYLOAD["commit"]
for row in PAYLOAD["parent_sources"]:
    assert hashlib.sha256((root / row["path"]).read_bytes().replace(
        b"\r\n", b"\n")).hexdigest() == row["lf_sha256"]
directory = root.parent / "pipeline/diagnostic_runtime" / P["token"] / "revision1" / PAYLOAD["package"]
directory.mkdir(parents=True, exist_ok=True)
for name, row in PAYLOAD["files"].items():
    raw = base64.b64decode(row["base64"], validate=True)
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    destination = directory / name
    if destination.exists():
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == row["sha256"]
    else:
        with destination.open("xb") as stream:
            stream.write(raw)
cfg = gorilla.Config.fromfile(str(directory / PAYLOAD["config_name"]))
assert yaml.safe_load((directory / PAYLOAD["config_name"]).read_text()) == PAYLOAD["config_values"]
assert cfg.train.seed == 1999 and cfg.train.epochs == 30
assert cfg.dataloader.train.batch_size == 2 and cfg.model.sampling_module.num_proposal == 128
namespace = "ipdn.model_paper5." + PAYLOAD["package"]
assert namespace not in sys.modules
spec = importlib.util.spec_from_file_location(namespace, str(directory / "__init__.py"),
                                            submodule_search_locations=[str(directory)])
package = importlib.util.module_from_spec(spec)
sys.modules[namespace] = package
spec.loader.exec_module(package)
Candidate = importlib.import_module(namespace + ".ipdn").IPDN
Indicator = importlib.import_module(namespace + ".dec").QueryCompetitionIndicator
torch.set_num_threads(2)
gorilla.set_random_seed(1999)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def initialize():
    assert P["physical_gpu"] in (0, 1) and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    gorilla.set_random_seed(1999)
    value = Candidate(**cfg.model).cuda()
    source = (root / "tools/train.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "load_shape_compatible_pretrain")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<exact-native-loader>", "exec"), globals())
    load_shape_compatible_pretrain(value, cfg.train.pretrain, get_root_logger())
    value.train()
    return value


def batches():
    dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
    items = []
    for scene in PAYLOAD["scenes"]:
        gorilla.set_random_seed(1999)
        item = dataset[scene["dataset_index"]]
        single = dataset.collate_fn([copy.deepcopy(item)])
        assert single["scan_ids"][0] == scene["scan_id"]
        assert [str(v) for v in single["ann_ids"]] == scene["ann_ids"]
        items.append(item)
    return (dataset, items,
            [dataset.collate_fn(copy.deepcopy(items[i:i + 2])) for i in range(0, len(items), 2)])


def preserved():
    for name, row in PAYLOAD["files"].items():
        assert digest(directory / name) == row["sha256"]
    assert (root / ".codex_source_commit").read_text().strip() == PAYLOAD["commit"]
    return True
