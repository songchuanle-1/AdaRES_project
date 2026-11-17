"""Mechanical parent copy and fixed final-readout insertion under this lease."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "e636ce91a12e45488093b07fa2fc092b"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
folder = parent.parent / "adares_y37_pair_distribution"
assert not folder.exists()
folder.mkdir()
for path in parent.glob("*.py"):
    shutil.copyfile(path, folder / path.name)
source = (parent / "dec.py").read_text(encoding="utf-8")
readout = (ROOT / "scripts/y37_pair_readout.py").read_text(encoding="utf-8")
node = next(n for n in ast.parse(readout).body if isinstance(n, ast.ClassDef))
assert source.count("class DEC(nn.Module):") == 1
source = source.replace("class DEC(nn.Module):",
                        ast.get_source_segment(readout, node) + "\n\n\nclass DEC(nn.Module):")
anchor = "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)"
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + "\n        with torch.random.fork_rng(devices=[]):\n"
                        "            self.pair_readout = MaskPairDistributionReadout(d_model)")
anchor = "        return {\n            'masks': pred_masks,"
assert source.count(anchor) == 1
source = source.replace(anchor, "        pred_indis = pred_indis + self.pair_readout(\n"
                        "            query, pred_masks, batch_mask, point_mass, inst_feats, sp_pos)\n\n" + anchor)
compile(source, str(folder / "dec.py"), "exec")
(folder / "dec.py").write_text(source, encoding="utf-8", newline="\n")
configs = ROOT / "AdaRES/configs_song/model_paper5"
target = configs / "adares_y37_pair_distribution_res.yaml"
assert not target.exists()
shutil.copyfile(configs / "adares_w2_pointmass_competition_res.yaml", target)
files = list(folder.glob("*.py")) + [target]
result = dict(complete=True, id="AdaRES-Y37", final_only=True,
              files=[dict(path=str(p.relative_to(ROOT)).replace("\\", "/"),
                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
output = ROOT / ("pipeline/analyses/y37_local_bundle_" + TOKEN + ".json")
assert not output.exists()
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
