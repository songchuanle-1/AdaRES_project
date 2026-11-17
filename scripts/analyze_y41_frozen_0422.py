"""FIRST-pass frozen whole-scene inference; no best-repeat selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
from analyze_y27_frozen_terminal import comparison
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--token', required=True)
args = parser.parse_args()
owner = json.loads((ROOT / 'pipeline/headless/.foreground.lockdir/owner.json').read_text())
assert owner['token'] == args.token and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner['deadline_at'])
path = ROOT / ('pipeline/analyses/y41_frozen_full_0422_' + args.token + '.json')
value = json.loads(path.read_text())
captures = value['rows']
assert value['complete'] and set(captures) == {'y41', 'w2', 'b1'}
assert all((c['complete'] and c['completed_passes'] == 2 for c in captures.values()))
result = {'complete': True, 'id': 'AdaRES-Y41', 'source': {'path': path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, 'observed_at_local': datetime.now().astimezone().isoformat(), 'primary': 'FIRST full pass, not favorable repeat', 'bootstrap': {'draws': 10000, 'seed': 1999, 'unit': 'whole scene'}, 'summary': {k: c['passes'][0]['summary'] for k, c in captures.items()}, 'repeated_miou': {k: c['passes'][1]['summary']['miou'] for k, c in captures.items()}, 'versus_w2': comparison(captures['y41'], captures['w2']), 'versus_b1': comparison(captures['y41'], captures['b1']), 'target': 0.515, 'target_reached': captures['y41']['passes'][0]['summary']['miou'] >= 0.515, 'cost_first_chunk_only': {k: {'parameters': c['parameters'], 'median_seconds': statistics.median(c['warm_first_chunk_seconds']), 'peak_allocated_gib': c['peak_allocated_gib'], 'peak_reserved_gib': c['peak_reserved_gib'], 'repeat_max_iou_difference': c['repeat_max_iou_difference'], 'repeat_miou_difference': c['repeat_miou_difference']} for k, c in captures.items()}, 'limitations': ['Official unique/multiple metadata unknown.', 'Two repeats are empirical noise, not training-seed uncertainty.', 'First-chunk cost is not trained paired Pareto evidence.', 'No novelty, upstream blindness, equivalence, necessity or no-harm claim.']}
output = ROOT / ('pipeline/analyses/y41_clustered_comparison_' + args.token + '.json')
assert not output.exists()
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: result[key] for key in ('summary', 'repeated_miou', 'versus_w2', 'versus_b1', 'target_reached')}, indent=2))
