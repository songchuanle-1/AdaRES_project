"""Bounded native main test with an isolated copy of the proposed entry."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import gorilla
import torch

from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_y17_target_equivalent_pairing.pairing import (
    encoded_pairs, target_ids)
from probe_adares_y17_native import Batches


class Completed(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([1], 1)
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("y17_proposed_train", args.entry)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    config = "configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml"
    cfg = gorilla.Config.fromfile(config)
    calls = []
    original = entry.build_dataset
    marker = object()
    entry.build_dataset = lambda *a: calls.append(a) or marker
    for name in ("adares_w2_pointmass_competition", "adares_y15_query_completion"):
        assert entry.resolve_train_dataset(
            "configs_song/model_paper5/"+name+"_res.yaml", cfg.data.train, None) is marker
    entry.build_dataset = original
    from ipdn.utils import get_root_logger
    dataset = entry.resolve_train_dataset(config, cfg.data.train, get_root_logger())
    assert dataset.__class__.__name__ == "PairedTrainingDataset"
    pair_count = 0
    chunks_with_pairs = 0
    for chunk in dataset.scene_inputs:
        texts = [" ".join(row["token"]) for row in chunk]
        tokens = dataset.tokenizer.batch_encode_plus(
            texts, padding="longest", return_tensors="pt")["input_ids"]
        pairs = encoded_pairs([target_ids(row) for row in chunk], tokens, [len(chunk)])[0]
        pair_count += len(pairs)
        chunks_with_pairs += bool(pairs)
    result = {"fallback_calls": len(calls), "fallback_identity": True,
              "encoded_pair_count": pair_count, "paired_expressions": pair_count*2,
              "chunks_with_pairs": chunks_with_pairs, "chunks": len(dataset),
              "epochs": cfg.train.epochs, "seed": cfg.train.seed,
              "native_main_complete": False}
    Path(args.output).write_text(json.dumps(result, indent=2))
    old_fromfile = entry.gorilla.Config.fromfile
    def diagnostic_cfg(path):
        value = copy.deepcopy(cfg)
        value.work_dir = "/bak_home/songcl/Project/AdaRES_project/y17_entry_diagnostic"
        return value
    entry.gorilla.Config.fromfile = diagnostic_cfg
    real_train = entry.train
    def bounded_train(epoch, model, loader, optimizer, scheduler, cfg, logger, writer):
        assert loader.dataset.__class__.__name__ == "PairedTrainingDataset"
        iterator = iter(loader)
        batches = [next(iterator), next(iterator)]
        assert all("target_pairs" in batch for batch in batches)
        real_train(epoch, model, Batches(batches), optimizer, scheduler, cfg, logger, writer)
        result["native_main_complete"] = True
        result["native_optimizer_steps"] = 2
        result["last_pair_count"] = model.last_pair_count
        Path(args.output).write_text(json.dumps(result, indent=2))
        writer.close()
        raise Completed()
    entry.train = bounded_train
    try:
        entry.main(SimpleNamespace(
            config=config, dist=False, num_gpus=1, resume=None,
            skip_training=False, skip_validate=False, save_ckpt=False))
    except Completed:
        pass
    finally:
        entry.gorilla.Config.fromfile = old_fromfile
    assert result["native_main_complete"]
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
