"""Specialize frozen existing probes using parsed constants, preserving old evidence."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])


class Constants(ast.NodeTransformer):
    def __init__(self, replacements):
        self.replacements = replacements

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = self.replacements.get(node.value, node.value)
            return ast.copy_location(ast.Constant(value=value), node)
        return node


specifications = [
    ("upload_y42_test_bundle.py", "upload_y43_test_bundle_0422.py", {
        "adares_y42_point_covariance": "adares_y43_point_asymmetry",
        "MaskPointCovarianceReadout": "MaskPointAsymmetryReadout",
        "AdaRES/configs_song/model_paper5/adares_y42_point_covariance_res.yaml":
            "AdaRES/configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml",
        "pipeline/analyses/y42_source_bundle_": "pipeline/analyses/y43_source_bundle_",
        "79ce2de9582c369422facb27c3cbe85559bf227c": "243e0b6827073831e0d68090857755d575efbf16",
        "AdaRES-Y42": "AdaRES-Y43",
    }),
    ("probe_y42_physical_native.py", "probe_y43_physical_native_0422.py", {
        "79ce2de9582c369422facb27c3cbe85559bf227c": "243e0b6827073831e0d68090857755d575efbf16",
        "configs_song/model_paper5/adares_y42_point_covariance_res.yaml":
            "configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml",
        "AdaRES-Y42": "AdaRES-Y43",
    }),
    ("probe_y42_resources.py", "probe_y43_resources_0422.py", {
        "79ce2de9582c369422facb27c3cbe85559bf227c": "243e0b6827073831e0d68090857755d575efbf16",
        "configs_song/model_paper5/adares_y42_point_covariance_res.yaml":
            "configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml",
        "AdaRES-Y42": "AdaRES-Y43",
    }),
    ("probe_y39_trained_cost_current.py", "probe_y41_trained_cost_0422.py", {
        "/proc/116906/cmdline": "/proc/91215/cmdline",
        "configs_song/model_paper5/adares_y41_boundary_probability_res.yaml":
            "configs_song/model_paper5/adares_y42_point_covariance_res.yaml",
        "/tmp/adares_y39_frozen_full_current_": "/tmp/adares_y41_frozen_full_0422_",
        "/tmp/adares_y39_terminal_current_": "/tmp/adares_y41_terminal_0422_",
        "y39": "y41", "AdaRES-Y39": "AdaRES-Y41", "0": "1",
    }),
    ("analyze_y39_frozen_current.py", "analyze_y41_frozen_0422.py", {
        "pipeline/analyses/y39_frozen_full_current_": "pipeline/analyses/y41_frozen_full_0422_",
        "pipeline/analyses/y39_clustered_comparison_": "pipeline/analyses/y41_clustered_comparison_",
        "y39": "y41", "AdaRES-Y39": "AdaRES-Y41",
    }),
    ("analyze_y39_object_groups.py", "analyze_y41_object_groups_0422.py", {
        "pipeline/analyses/y39_frozen_full_current_": "pipeline/analyses/y41_frozen_full_0422_",
        "pipeline/analyses/y39_object_groups_": "pipeline/analyses/y41_object_groups_",
        "y39": "y41", "AdaRES-Y39": "AdaRES-Y41",
    }),
]
records = []
for source_name, target_name, replacements in specifications:
    source = ROOT / "scripts" / source_name
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree = Constants(replacements).visit(tree)
    # The upload's remote embedded program is one string, so bind its literal paths explicitly.
    if source_name == "upload_y42_test_bundle.py":
        program = next(n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "program" for t in n.targets))
        text = program.value.value
        assert text.count("adares_y42_point_covariance") == 2
        program.value = ast.Constant(value=text.replace("adares_y42_point_covariance", "adares_y43_point_asymmetry"))
    ast.fix_missing_locations(tree)
    target = ROOT / "scripts" / target_name
    assert not target.exists()
    text = ast.unparse(tree) + "\n"
    compile(text, str(target), "exec")
    target.write_text(text, encoding="utf-8", newline="\n")
    records.append({"path": target.relative_to(ROOT).as_posix(),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "parent_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
output = ROOT / ("pipeline/analyses/bound_protocols_0422_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps({"complete": True, "files": records,
    "limits": ["Bindings only, unexecuted tests remain pending."]}, indent=2), encoding="utf-8")
print(json.dumps({"files": len(records), "path": str(output)}))
