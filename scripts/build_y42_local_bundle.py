"""Mechanical isolated W2 copy with full covariance, not a channel subset."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
target = parent.parent / "adares_y42_point_covariance"
if target.exists():
    copied = sorted(target.glob("*.py"))
    assert len(copied) == 6
    assert all(p.read_bytes() == (parent / p.name).read_bytes() for p in copied)
else:
    target.mkdir()
    for path in parent.glob("*.py"):
        shutil.copyfile(path, target / path.name)
readout = (ROOT / "scripts/y42_covariance_readout.py").read_text(encoding="utf-8")
node = next(n for n in ast.parse(readout).body if isinstance(n, ast.ClassDef))
dec = (parent / "dec.py").read_text(encoding="utf-8")
assert dec.count("class DEC(nn.Module):") == 1
dec = dec.replace("class DEC(nn.Module):", ast.get_source_segment(readout, node)
                  + "\n\n\nclass DEC(nn.Module):")
anchor = "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)"
assert dec.count(anchor) == 1
dec = dec.replace(anchor, anchor + "\n        with torch.random.fork_rng(devices=[]):\n"
                  "            self.point_readout = MaskPointCovarianceReadout(d_model, in_channel)")
signature = "feats_2d=None, point_mass=None):"
assert dec.count(signature) == 2
dec = dec.replace(signature, "feats_2d=None, point_mass=None, point_covariance=None):")
anchor = "        point_mass = point_mass.squeeze(-1)"
assert dec.count(anchor) == 1
dec = dec.replace(anchor, anchor + "\n        if point_covariance is None:\n"
                  "            raise ValueError('Native point covariance is required')\n"
                  "        point_covariance, _ = self.get_batches(point_covariance, batch_offsets)")
anchor = "        return {\n            'masks': pred_masks,"
assert dec.count(anchor) == 1
dec = dec.replace(anchor, "        pred_indis = pred_indis + self.point_readout(\n"
                  "            query, pred_masks, batch_mask, point_mass, point_covariance)\n\n" + anchor)
anchor = "lang_masks, sp_pos, feats_2d, point_mass)"
assert dec.count(anchor) == 1
dec = dec.replace(anchor, "lang_masks, sp_pos, feats_2d, point_mass, point_covariance)")
compile(dec, str(target / "dec.py"), "exec")
(target / "dec.py").write_text(dec, encoding="utf-8", newline="\n")

ipdn = (parent / "ipdn.py").read_text(encoding="utf-8")
anchor = "        sp_feats = self.extract_feat(input, superpoints, p2v_map)"
assert ipdn.count(anchor) == 2
ipdn = ipdn.replace(anchor, "        sp_feats, point_covariance = self.extract_feat(input, superpoints, p2v_map)\n"
                    "        point_covariance = torch.cat([\n"
                    "            point_covariance[int(batch_offsets[i]):int(batch_offsets[i+1])].repeat(int(scenes_len[i]), 1)\n"
                    "            for i in range(len(scenes_len))\n"
                    "        ])")
anchor = "lang_masks, sp_coords_float, feats_2d, point_mass)"
assert ipdn.count(anchor) == 2
ipdn = ipdn.replace(anchor, "lang_masks, sp_coords_float, feats_2d, point_mass, point_covariance)")
anchor = "        # superpoint pooling"
assert ipdn.count(anchor) == 1
ipdn = ipdn.replace(anchor,
                    "        mean_features = scatter_mean(x, superpoints, dim=0)\n"
                    "        centered = x - mean_features[superpoints]\n"
                    "        # Stream rows to avoid a point-by-channel-by-channel tensor.\n"
                    "        joint_rows = [\n"
                    "            scatter_mean(centered[:, i:i+1] * centered[:, i:], superpoints, dim=0)\n"
                    "            for i in range(centered.shape[1])\n"
                    "        ]\n"
                    "        point_covariance = torch.cat(joint_rows, dim=1)\n\n" + anchor)
anchor = "        return x\n"
assert ipdn.count(anchor) == 1
ipdn = ipdn.replace(anchor, "        return x, point_covariance\n")
compile(ipdn, str(target / "ipdn.py"), "exec")
(target / "ipdn.py").write_text(ipdn, encoding="utf-8", newline="\n")
config_dir = ROOT / "AdaRES/configs_song/model_paper5"
config = config_dir / "adares_y42_point_covariance_res.yaml"
assert not config.exists()
shutil.copyfile(config_dir / "adares_w2_pointmass_competition_res.yaml", config)
files = sorted(target.glob("*.py")) + [config]
assert len(files) == 7
output = ROOT / ("pipeline/analyses/y42_local_bundle_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps({
    "complete": True, "id": "AdaRES-Y42", "final_only": True,
    "files": [{"path": p.relative_to(ROOT).as_posix(),
               "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
}, indent=2), encoding="utf-8")
print(json.dumps({"files": len(files), "path": str(output)}))
