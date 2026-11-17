"""Preserve the tested checker bytes before a bounded operational repair."""

import hashlib
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']=='f27d10d52a2b4fabadfd1c28afd46c65' and owner['thread_id']==os.environ['CODEX_THREAD_ID']
source=ROOT/'scripts/check_ef_execution.py'
target=ROOT/'pipeline/analyses/check_ef_execution_f27d_v1.py'
assert not target.exists()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copyfile(source,target)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
    work=tick.load(ROOT/'pipeline/research_work.json')
    changes=0
    for c in work['candidates']:
        for row in c['evidence']:
            if row['path']=='scripts/check_ef_execution.py':
                assert row['sha256']==digest
                row['path']=target.relative_to(ROOT).as_posix()
                changes+=1
    assert changes==2
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Preserved the old checker and both exact hash bindings before repair; no scientific result removed.')
finally:
    lock.close()
