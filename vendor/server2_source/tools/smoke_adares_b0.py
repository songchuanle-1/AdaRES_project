import argparse
import importlib
import os
from pathlib import Path
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
        description="AdaRES real-batch smoke test."
    )
    parser.add_argument("config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1999)
    parser.add_argument(
        "--model-module",
        default="ipdn",
        help="Module that exports IPDN, e.g. ipdn.model_paper5.variant.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(args.seed)
    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = cfg.data.train.copy()
    data_cfg.training = False
    data_cfg.aug = False
    data_cfg.with_elastic = False
    data_cfg.lang_num_max = 2
    logger = get_root_logger(
        log_file=os.path.join(args.output_dir, "smoke.log")
    )
    dataset = build_dataset(data_cfg, logger)
    model_module = importlib.import_module(args.model_module)
    IPDN = model_module.IPDN
    model = IPDN(**cfg.model)
    model = model.cuda()
    model.train()

    pretrain = cfg.train.get("pretrain", None)
    if pretrain and os.path.isfile(pretrain):
        checkpoint = torch.load(pretrain, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        own = model.state_dict()
        filtered = {
            key: value
            for key, value in state.items()
            if key in own and own[key].shape == value.shape
        }
        result = model.load_state_dict(filtered, strict=False)
        logger.info(
            "Pretrain loaded=%d missing=%d unexpected=%d",
            len(filtered),
            len(result.missing_keys),
            len(result.unexpected_keys),
        )

    losses = []
    gradients = []
    started = time.perf_counter()
    for index in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[index]])
        loss, terms = model(batch, mode="loss")
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at dataset index {index}"
            )
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        losses.append(float(loss.detach().item()))
        gradients.append(float(gradient.detach().item()))
        detached_terms = {
            key: round(float(value.detach().item()), 4)
            for key, value in terms.items()
        }
        logger.info(
            "scene=%d loss=%.6f grad=%.6f",
            index,
            losses[-1],
            gradients[-1],
        )
        logger.info("loss terms: %s", detached_terms)

    model.eval()
    batch = dataset.collate_fn([dataset[0]])
    with torch.no_grad():
        prediction = model(batch, mode="predict")
    result = {
        "losses": losses,
        "gradient_norms": gradients,
        "predict_keys": sorted(prediction.keys()),
        "piou": float(prediction["piou"][0].item()),
        "spiou": float(prediction["spiou"][0].item()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    logger.info("Smoke result: %s", result)
    print(result)


if __name__ == "__main__":
    main()
