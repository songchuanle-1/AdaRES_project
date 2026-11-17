"""Freeze component tests and exact source configuration, not a candidate model."""

import ast
from datetime import datetime
import subprocess
from research_b780 import ROOT,TOKEN,lease,ref,save,bind,tick,monitor

lease()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y58')
    parent=ROOT/'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
    tree=ast.parse((parent/'ipdn.py').read_text(encoding='utf-8'))
    bn=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='partial' and n.args and isinstance(n.args[0],ast.Attribute) and n.args[0].attr=='BatchNorm1d']
    assert len(bn)==1
    options={n.arg:ast.literal_eval(n.value) for n in bn[0].keywords}
    assert options==dict(eps=1e-4,momentum=0.1)
    old=ROOT/'pipeline/analyses/y56_native_batch2_222ed2a6e80c43f6b1de2d050be97fa0_program.py'
    frozen=ast.literal_eval(next(n.value for n in ast.parse(old.read_text(encoding='utf-8')).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FROZEN' for t in n.targets)))
    code=ast.parse((parent/'dec.py').read_text(encoding='utf-8'))
    head=ast.unparse(next(n for n in code.body if isinstance(n,ast.ClassDef) and n.name=='QueryCompetitionIndicator'))
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert commit=='05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea'
    payload=dict(commit=commit,sources=frozen['sources'],checkpoint=frozen['checkpoint'],
        checkpoint_sha256=frozen['checkpoint_sha256'],parent_class=head,bn_eps=options['eps'],bn_momentum=options['momentum'])
    body=ROOT/'scripts/probe_y58_cpu_b780.py'
    program='PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8')
    ast.parse(program,feature_version=(3,8))
    path='pipeline/analyses/y58_cpu_'+TOKEN+'_program.py'
    assert not (ROOT/path).exists()
    (ROOT/path).write_text(program,encoding='utf-8',newline='\n')
    related=[dict(id=row['id'],stage=row['stage'],hypothesis_id=row.get('hypothesis_id'))
        for row in work['candidates'] if any(word in (row.get('hypothesis_id','')+' '+row.get('hypothesis','')).lower()
            for word in ('normalization','batchnorm','batch_norm','batch mate','cross_scene','running_stat'))]
    source=save('pipeline/analyses/y58_source_review_'+TOKEN+'.json',dict(complete=True,id='AdaRES-Y58',
        observed_at=datetime.now().astimezone().isoformat(),BN_source_options=options,
        source=[ref(parent/'ipdn.py'),ref(parent/'backbone.py'),ref(parent/'dec.py')],
        train_norm_eval=False,indicator_has_batch_mixing=False,
        targeted_related_records=related,
        interpretation='No matching cross-scene backbone BN mechanism found in the targeted ledger fields. '
                       'Y23 padding denominator and Y41 boundary normalization are different, not reopened. '
                       'This is not a literature novelty proof. Actual native task relevance remains pending.'))
    plan=save(path.replace('_program.py','_prereg.json'),dict(complete=True,id='AdaRES-Y58',
        preregistered_at=datetime.now().astimezone().isoformat(),program=ref(path),body=ref(body),source_review=source,
        protocol='Original seed1999; real output_layer.0 trained affine/running stats, float32/64, focal32/mate48 '
                 'synthetic features, pooled-vs-separated/fixed-stat/repeat/permutation; actual trained W2 head separability.',
        result='pending',no_native_or_admission_claim=True))
    bind(c,ref(path),plan,source,ref(body),ref(__file__))
    c['next_action']='Execute frozen CPU component and actual-trained-head negatives now; do not infer native drift or change any running experiment.'
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Frozen Y58 source-derived BN and actual-trained-head CPU controls.')
finally:
    lock.close()
