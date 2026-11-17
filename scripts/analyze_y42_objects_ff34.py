"""Supplement actual object-count strata without fabricating absent metadata."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from analyze_y23_frozen_terminal import comparison
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--token', required=True)
args = parser.parse_args()
owner = json.loads((ROOT / 'pipeline/headless/.foreground.lockdir/owner.json').read_text())
assert owner['token'] == args.token and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner['deadline_at'])
source = ROOT / ('pipeline/analyses/y42_frozen_full_ff34_' + args.token + '.json')
capture = json.loads(source.read_text())['rows']
groups = {}
for name, condition in (('single_object', lambda row: len(row['object_ids']) == 1), ('multi_object', lambda row: len(row['object_ids']) > 1)):
    subsets = {}
    for model, value in capture.items():
        changed = copy.deepcopy(value)
        for iteration in changed['passes']:
            iteration['rows'] = [row for row in iteration['rows'] if condition(row)]
        subsets[model] = changed
    rows = subsets['y42']['passes'][0]['rows']
    scene_count = len({r['scan_id'] for r in rows})
    if scene_count < 2:
        groups[name] = {'expressions': len(rows), 'scenes': scene_count, 'status': 'insufficient population; no CI/harm/no-harm assertion'}
    else:
        groups[name] = {reference: comparison(subsets['y42'], subsets[reference])['all'] for reference in ('w2', 'b1')}
        for result in groups[name].values():
            result['frozen_material_harm_ci_upper_below_zero'] = result['scene_clustered_ci95'][1] < 0
result = {'complete': True, 'id': 'AdaRES-Y42', 'groups': groups, 'source': {'path': source.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}, 'limits': ['Object-count strata are not official unique/multiple categories.', 'An absent multi-object population is not evidence of safety.', 'Original full-population and size-stratum comparison is unchanged.']}
output = ROOT / ('pipeline/analyses/y42_object_groups_' + args.token + '.json')
assert not output.exists()
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result['groups'], indent=2))
