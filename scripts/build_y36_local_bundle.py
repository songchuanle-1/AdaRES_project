"""Mechanically copy the parent and insert the fixed candidate-only path."""

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "8fd5e469dee241da8f21da7c870c0d3f"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
folder = parent.parent / "adares_y36_local_adjacency"
assert not folder.exists()
folder.mkdir()
for path in parent.glob("*.py"):
    shutil.copyfile(path, folder / path.name)
source = (parent / "dec.py").read_text(encoding="utf-8")
readout = (ROOT / "scripts/y36_edge_readout.py").read_text(encoding="utf-8")
class_node = next(n for n in ast.parse(readout).body if isinstance(n, ast.ClassDef))
class_source = ast.get_source_segment(readout, class_node)
assert source.count("class DEC(nn.Module):") == 1
source = source.replace("class DEC(nn.Module):", class_source + "\n\n\nclass DEC(nn.Module):")
init_anchor = "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)"
assert source.count(init_anchor) == 1
source = source.replace(init_anchor, init_anchor + "\n"
    "        with torch.random.fork_rng(devices=[]):\n"
    "            self.edge_readout = MaskLocalAdjacencyReadout(d_model)")
final_anchor = "        return {\n            'masks': pred_masks,"
assert source.count(final_anchor) == 1
source = source.replace(final_anchor,
    "        pred_indis = pred_indis + self.edge_readout(\n"
    "            query, pred_masks, batch_mask, point_mass, inst_feats, sp_pos)\n\n" + final_anchor)
compile(source, str(folder / "dec.py"), "exec")
(folder / "dec.py").write_text(source, encoding="utf-8", newline="\n")
config_dir = ROOT / "AdaRES/configs_song/model_paper5"
target = config_dir / "adares_y36_local_adjacency_res.yaml"
assert not target.exists()
shutil.copyfile(config_dir / "adares_w2_pointmass_competition_res.yaml", target)
print(json.dumps(dict(complete=True, candidate="AdaRES-Y36",
                      files=6, added_class=class_node.name, final_only=True)))
