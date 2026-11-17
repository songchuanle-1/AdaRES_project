"""Freeze source-bound native data identity controls before execution."""

import ast
from datetime import datetime
from research_b780 import ROOT,TOKEN,lease,ref,save,bind,tick,monitor
lease()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y58')
    old=ROOT/'pipeline/analyses/y56_native_batch2_222ed2a6e80c43f6b1de2d050be97fa0_program.py'
    frozen=ast.literal_eval(ast.parse(old.read_text(encoding='utf-8')).body[0].value)
    frozen['commit']='05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea'
    body=ROOT/'scripts/probe_y58_data_b780.py'
    program='FROZEN='+repr(frozen)+'\n'+body.read_text(encoding='utf-8')
    ast.parse(program,feature_version=(3,8))
    path='pipeline/analyses/y58_data_'+TOKEN+'_program.py'
    assert not (ROOT/path).exists()
    (ROOT/path).write_text(program,encoding='utf-8',newline='\n')
    plan=save(path.replace('_program.py','_prereg.json'),dict(complete=True,id='AdaRES-Y58',
        preregistered_at=datetime.now().astimezone().isoformat(),program=ref(path),body=ref(body),
        sample='First3 of original fixed8TRAIN scenes: focal0, mates1/2, repeatmate1; seed1999 native augmentation.',
        prediction='Raw focal points/features/GT/annotation and language tensors remain bitwise identical across batch mates.',
        source_binding='Record actual installed dataset class/file hash and collate hash for the future native test.',
        native_model_executed=False,result='pending'))
    bind(c,ref(path),plan,ref(body),ref(__file__),ref('pipeline/analyses/y58_cpu_'+TOKEN+'.json'),
         ref('pipeline/analyses/y58_cpu_'+TOKEN+'.log'))
    c['next_action']='Execute native dataset CPU identity controls, then freeze exact future exclusive native BN causality test; no new training.'
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Frozen actual TRAIN data identity negative controls.')
finally:
    lock.close()
