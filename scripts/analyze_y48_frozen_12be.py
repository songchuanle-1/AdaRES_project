"""Apply the existing frozen whole-scene and subgroup comparisons to Y48."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from analyze_y27_frozen_terminal import comparison
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--token', required=True)
args = parser.parse_args()
owner = json.loads((ROOT / 'pipeline/headless/.foreground.lockdir/owner.json').read_text())
assert owner['token'] == args.token and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner['deadline_at'])
source = ROOT / ('pipeline/analyses/y48_frozen_12be_' + args.token + '.json')
data = json.loads(source.read_text(encoding='utf-8'))
captures = data['rows']
assert data['complete'] and set(captures) == {'y48', 'w2', 'b1'}
assert data['newly_executed_full_passes'] == 6
for capture in captures.values():
    assert capture['complete'] and capture['completed_passes'] == 2
    assert all((p['summary']['expressions'] == 9508 and p['summary']['scenes'] == 141 for p in capture['passes']))
result = dict(complete=True, id='AdaRES-Y48', observed_at_local=datetime.now().astimezone().isoformat(), source=dict(path=source.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(source.read_bytes()).hexdigest()), primary='FIRST full pass; no favorable repeat or validation retuning', bootstrap=dict(draws=10000, seed=1999, unit='whole scene'), summary={key: value['passes'][0]['summary'] for key, value in captures.items()}, repeated_miou={key: value['passes'][1]['summary']['miou'] for key, value in captures.items()}, versus_w2=comparison(captures['y48'], captures['w2']), versus_b1=comparison(captures['y48'], captures['b1']), target=0.515, target_reached=captures['y48']['passes'][0]['summary']['miou'] >= 0.515, repeat_max_iou_difference={key: value['repeat_max_iou_difference'] for key, value in captures.items()}, active_queries=128, selected_queries=1, newly_executed_full_passes=6, limitations=['Official unique/multiple metadata unknown; no substitute object-count claim.', 'One trained seed per method; repeats are not training-seed uncertainty.', 'No trained paired cost claim until separate native measurement.', 'A negative realization does not automatically falsify the broader mechanism.'])
result['frozen_harm_triggered_groups'] = {ref: [group for group, value in result['versus_' + ref].items() if value.get('frozen_material_harm_ci_upper_below_zero')] for ref in ('w2', 'b1')}
destination = ROOT / ('pipeline/analyses/y48_clustered_12be_' + args.token + '.json')
with destination.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({key: result[key] for key in ('summary', 'repeated_miou', 'versus_w2', 'versus_b1', 'target_reached', 'frozen_harm_triggered_groups')}, indent=2))
