"""Freeze one genuinely new CPU construction control under the live lease."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
parser=argparse.ArgumentParser()
parser.add_argument('--token',required=True)
args=parser.parse_args()
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==args.token and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds()>600
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
bundle_path='pipeline/analyses/y52_prototype_bundle_4567_4567f5a09ecb4aa89df8f0c2932652df.json'
bundle=tick.load(ROOT/bundle_path)
payload=dict(python_files={})
for row in bundle['files']:
    assert ref(row['path'])==row
    value=dict(base64=base64.b64encode((ROOT/row['path']).read_bytes()).decode(),sha256=row['sha256'])
    if row['path'].endswith('.py'):payload['python_files'][row['path'].rsplit('/',1)[-1]]=value
    else:payload['config']=value
assert len(payload['python_files'])==6
body_path='scripts/probe_y52_full_cpu_construction.py'
source='PAYLOAD='+repr(payload)+'\n'+(ROOT/body_path).read_text(encoding='utf-8')
program_path='pipeline/analyses/y52_cpu_construction_program_'+args.token+'.py'
prereg_path='pipeline/analyses/y52_cpu_construction_prereg_'+args.token+'.json'
assert not (ROOT/program_path).exists() and not (ROOT/prereg_path).exists()
compile(source,program_path,'exec')
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    tick.save(ROOT/prereg_path,dict(complete=True,id='AdaRES-Y52',token=args.token,
        preregistered_at=datetime.now().astimezone().isoformat(),prototype=ref(bundle_path),body=ref(body_path),
        test='Physically import the exact six independent files from a temporary namespace, construct actual full IPDN on CPU at seed1999, and compare all parameter counts/state shapes and exact initial tensors with actual W2 under identical constructor configuration.',
        purpose='Exercise full-package imports/constructor integration not covered by prior loss-only CPU imports; SAME candidate, not an implementation revision.',
        prediction='Independent package constructs with zero added parameters and exact fixed-seed parent initialization; mismatch is an implementation or instrumentation failure to preserve, not automatic hypothesis falsification.',
        scope='CUDA hidden; no forward/backward/training, no released-backbone training, no model/checkpoint weights saved or deployed. Native TRAIN/backward/resource/restore gates remain pending.',
        timeout_seconds=180))
    (ROOT/program_path).write_text(source,encoding='utf-8',newline='\n')
    print(json.dumps(dict(preregistration=ref(prereg_path),program=ref(program_path)),indent=2))
finally:lock.close()
