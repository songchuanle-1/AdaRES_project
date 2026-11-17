import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import gorilla
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from diagnose_adares_w2_point_weighted_competition import reweight


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    counts = torch.bincount(
        superpoints.long(), minlength=out["masks"].shape[-1]
    ).to(device=out["masks"].device, dtype=out["masks"].dtype)
    manual = reweight(
        self.dec.indi_embedding, self.probe_query, out["masks"],
        out["batch_mask"], counts[None].expand(out["masks"].shape[0], -1)
    )
    native = out["indis"].softmax(-1)[..., 1]
    valid = ~out["batch_mask"]
    return {
        "max_score_difference": float((native - manual).abs().max()),
        "selected_equal": bool(torch.equal(native.argmax(-1), manual.argmax(-1))),
        "rows": native.shape[0],
        "padded_mass_zero": bool((self.probe_mass * ~valid).sum() == 0),
        "mass_sum": float(self.probe_mass.sum()),
        "point_count": int(superpoints.numel() * native.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = cfg.data.train.copy()
    data_cfg.training = False
    data_cfg.aug = False
    data_cfg.with_elastic = False
    data_cfg.lang_num_max = 32
    dataset = build_dataset(data_cfg, get_root_logger())
    import ipdn.model_paper5
    ipdn.model_paper5.__path__.append(str(args.package_dir.resolve()))
    module = importlib.import_module(
        "ipdn.model_paper5.adares_w2_pointmass_competition")
    model = module.IPDN(**cfg.model)
    model.cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    loaded = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    allowed = {"text_encoder.embeddings.position_ids",
               "text_encoder.embeddings.token_type_ids"}
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.eval()

    def capture(_, inputs):
        model.probe_query = inputs[0]
        model.probe_mass = inputs[3]

    model.dec.indi_embedding.register_forward_pre_hook(capture)
    model.predict_by_feat = types.MethodType(predictor, model)
    results = []
    for index in (0, 1):
        with torch.no_grad():
            result = model(dataset.collate_fn([dataset[index]]), mode="predict")
        assert result["selected_equal"] and result["max_score_difference"] < 1e-5
        assert result["padded_mass_zero"]
        assert result["mass_sum"] == result["point_count"]
        results.append(result)
    print(json.dumps({"scenes": results}))


if __name__ == "__main__":
    main()
