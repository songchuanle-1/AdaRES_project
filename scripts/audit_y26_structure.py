"""Prospective Y26 eligibility audit; preserve all previous raw evidence."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "11c9ed2af51a4ed9967596b52811d11d"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/dec.py"
candidate = parent.parent.parent / "adares_y26_qla_postnorm/dec.py"
old = parent.read_text(encoding="utf-8")
new = candidate.read_text(encoding="utf-8")
expected = old.replace("def __init__(self, d_model=256, nhead=8, dropout=0.0):",
                       "def __init__(self, d_model=256, nhead=8, dropout=0.0, postnorm=False):", 1)
expected = expected.replace("        self.nhead = nhead\n",
                            "        self.nhead = nhead\n        self.postnorm = postnorm\n", 1)
expected = expected.replace("        self.norm(output)\n",
                            "        normalized = self.norm(output)\n        if self.postnorm:\n            output = normalized\n", 1)
expected = expected.replace("self.qla_layers.append(CrossAttentionLayer(d_model, nhead, dropout))",
                            "self.qla_layers.append(CrossAttentionLayer(d_model, nhead, dropout, postnorm=True))", 1)
assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(new))
policy = Path("C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/references/autonomous_design.md")
out = ROOT / ("pipeline/analyses/y26_structure_" + TOKEN + ".json")
assert not out.exists()
result = {
    "complete": True, "id": "AdaRES-Y26", "observed_local": datetime.now().astimezone().isoformat(),
    "exact_ast_matches_only_qla_normalization_reconnection": True,
    "additional_trainable_parameters": 0, "new_nontrivial_information_interaction": False,
    "smell_signals": [1, 2, 3, 6], "admission_eligible": False,
    "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
    "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
    "decision": "Reject autonomous successor eligibility of norm-only realization; engineering diagnosis preserved.",
    "limitations": ["No trained Y26 effect was measured; not a harmful-efficacy rejection.",
                    "Previous six engineering gates omitted mandatory smell check; previous outputs remain immutable."],
}
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                  "result": result}, indent=2))
