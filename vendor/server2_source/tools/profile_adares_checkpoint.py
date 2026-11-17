import argparse
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import gorilla
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile an AdaRES checkpoint with a fixed protocol."
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--model-module", default="ipdn")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--lang-num-max", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1999)
    return parser.parse_args()


def effective_query_count(scores):
    scores = scores.float().clamp_min(0)
    weights = scores / scores.sum().clamp_min(1e-8)
    return float((1.0 / weights.square().sum().clamp_min(1e-8)).item())


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    result = model.load_state_dict(state, strict=False)
    allowed_unexpected = {
        "text_encoder.embeddings.position_ids",
        "text_encoder.embeddings.token_type_ids",
    }
    unexpected = set(result.unexpected_keys)
    if result.missing_keys or unexpected - allowed_unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )


def main():
    args = parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(args.seed)

    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = cfg.data.val.copy()
    data_cfg.training = False
    data_cfg.aug = False
    data_cfg.with_elastic = False
    data_cfg.lang_num_max = args.lang_num_max
    logger = get_root_logger(
        log_file=str(output_path.with_suffix(".log"))
    )
    dataset = build_dataset(data_cfg, logger)

    model_module = importlib.import_module(args.model_module)
    model = model_module.IPDN(**cfg.model)
    model = model.cuda()
    model.eval()
    load_checkpoint(model, args.checkpoint)

    latencies_ms = []
    active_queries = []
    pious = []
    spious = []
    expression_count = 0
    peak_memory_bytes = 0
    scene_count = min(args.scenes, len(dataset))

    for index in range(scene_count):
        batch = dataset.collate_fn([dataset[index]])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.no_grad():
            prediction = model(batch, mode="predict")
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        peak_memory_bytes = max(
            peak_memory_bytes, torch.cuda.max_memory_allocated()
        )

        count = len(prediction["piou"])
        if count == 0:
            continue
        expression_count += count
        latencies_ms.extend([elapsed_ms / count] * count)
        pious.extend(float(value.item()) for value in prediction["piou"])
        spious.extend(float(value.item()) for value in prediction["spiou"])
        active_queries.extend(
            effective_query_count(scores)
            for scores in prediction["indi"]
        )

    if expression_count == 0:
        raise RuntimeError("The profiling subset produced no expressions.")

    result = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_module": args.model_module,
        "seed": args.seed,
        "scenes": scene_count,
        "expressions": expression_count,
        "miou": statistics.fmean(pious),
        "spmiou": statistics.fmean(spious),
        "effective_active_queries_mean": statistics.fmean(active_queries),
        "effective_active_queries_std": (
            statistics.pstdev(active_queries)
            if len(active_queries) > 1
            else 0.0
        ),
        "latency_ms_per_expression_mean": statistics.fmean(latencies_ms),
        "latency_ms_per_expression_median": statistics.median(latencies_ms),
        "peak_memory_mib": peak_memory_bytes / (1024.0 * 1024.0),
        "active_query_definition": (
            "1/sum(w_q^2), w_q=s_q/sum_j(s_j), where s_q is the "
            "indicator positive-class probability"
        ),
    }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Checkpoint profile: %s", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
