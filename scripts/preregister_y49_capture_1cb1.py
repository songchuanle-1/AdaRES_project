"""Bind exact previously frozen cohort and new capture program before execution."""
import argparse,hashlib,os,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
parser=argparse.ArgumentParser()
parser.add_argument('--token',required=True)
parser.add_argument('--suffix',choices=('', 'v2'),default='')
args=parser.parse_args()
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==args.token and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc)<datetime.fromisoformat(owner['deadline_at'])
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
program=ref('scripts/probe_y49_train24_capture_1cb1'+('_v2' if args.suffix else '')+'.py')
cohort=ref('pipeline/analyses/y49_train_extension_freeze_f56b_f56bf800538f452ebf4b11f81ca8fd8c.json')
record=dict(complete=True,id='AdaRES-Y49',preregistered_at=datetime.now().astimezone().isoformat(),program=program,cohort=cohort,scope='Exactly the previously frozen24 new TRAIN scenes/first4 expressions; capture functions AST from original hash2ba5bad4. No rerun original TRAIN8/CPU32 controls.',test='Collect native inputs and passive TRAIN GT, then SAME fixed unit-resolvent/component-mass collision count under existing8eps*nodes bound. Preserve old0 and report any new0 without promotion.',no_revision_or_training=True)
relative='pipeline/analyses/y49_train24_capture_prereg_1cb1_'+args.suffix+args.token+'.json'
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    assert not (ROOT/relative).exists()
    tick.save(ROOT/relative,record)
    work_path=ROOT/'pipeline/research_work.json'
    work=tick.load(work_path)
    c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y49')
    assert c['stage']=='validation'
    c['evidence'].extend([ref(relative),program])
    c['next_action']='Execute exact frozen additionalTRAIN24 capture, then current production resolvent/component information diagnosis; old zero and runtime spikes retained, no admission from capture.'
    work.pop('design_cycle',None)
    tick.save(work_path,work)
    print(relative)
finally:lock.close()
