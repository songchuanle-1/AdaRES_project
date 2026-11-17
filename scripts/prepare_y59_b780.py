"""Register a separate frozen-language mode-contract diagnosis before testing."""

import ast
from datetime import datetime
from research_b780 import ROOT,TOKEN,lease,ref,save,bind,tick,monitor
lease()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    assert not any(c['id']=='AdaRES-Y59' for c in work['candidates'])
    parent=ROOT/'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py'
    tree=ast.parse(parent.read_text(encoding='utf-8'))
    model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IPDN')
    train=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    shim=ast.unparse(ast.ClassDef(name='IPDN',bases=[ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),
        attr='Module',ctx=ast.Load())],keywords=[],body=[train],decorator_list=[]))
    old=ROOT/'pipeline/analyses/y56_native_batch2_222ed2a6e80c43f6b1de2d050be97fa0_program.py'
    frozen=ast.literal_eval(ast.parse(old.read_text(encoding='utf-8')).body[0].value)
    frozen['commit']='05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea'
    payload=dict(frozen=frozen,exact_train_method_shim=shim,
        dataset=tick.load(ROOT/('pipeline/analyses/y58_data_'+TOKEN+'.json'))['dataset_source'])
    body=ROOT/'scripts/probe_y59_cpu_b780.py'
    program='PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8')
    ast.parse(program,feature_version=(3,8))
    path='pipeline/analyses/y59_cpu_'+TOKEN+'_program.py'
    assert not (ROOT/path).exists()
    (ROOT/path).write_text(program,encoding='utf-8',newline='\n')
    related=[dict(id=c['id'],stage=c['stage'],hypothesis_id=c.get('hypothesis_id')) for c in work['candidates']
        if any(word in (c.get('hypothesis_id','')+' '+c.get('hypothesis','')).lower() for word in
               ('dropout','roberta','frozen_language','frozen_text','frozen_encoder','mode_contract'))]
    assert not related,'Inspect an existing related route before registering a possibly duplicate ID'
    plan=save('pipeline/analyses/y59_prereg_'+TOKEN+'.json',dict(complete=True,id='AdaRES-Y59',
        preregistered_at=datetime.now().astimezone().isoformat(),hypothesis_id='frozen_text_encoder_mode_contract',
        hypothesis='The constructor explicitly puts fully frozen text_encoder into eval mode, but the exact '
                   'outer train() override may reactivate frozen encoder dropout. Check actual trained language '
                   'features and their native decoder projection, not assumed task benefit.',
        source=ref(parent),program=ref(path),body=ref(body),targeted_related_records=related,
        controls='Actual frozen trained W2 Roberta, native TRAIN tokens/mask, exact original IPDN.train method '
                 'in a CPU-only Module shim. Same-input uncontrolled-RNG repeat vs reset-RNG repeat; '
                 'constructor eval vs outer train followed by restoring encoder eval. Parent stays training.',
        readouts='Encoder mode/active dropout flags, frozen gradients, valid-token hidden and actual '
                 'trained dec.lang_proj drift. Same128eps arithmetic bound; no validation or dropout-rate search.',
        future_native='Original vs restored-frozen-mode independent package on same native TRAIN batches, '
                      'full loss/every confidence tensor/native resource guards before considering admission.',
        no_model_or_scheduler_changes=True,trained_efficacy='unknown',novelty='not claimed; potential correctness repair',
        result='pending'))
    c=dict(id='AdaRES-Y59',stage='diagnosis',created_at=datetime.now().astimezone().isoformat(),
        hypothesis_id='frozen_text_encoder_mode_contract',
        hypothesis='Outer train() may undo constructor eval for fully frozen text_encoder, injecting '
                   'unintended stochastic language evidence into confidence learning. Diagnose and preserve '
                   'a fixed-module mode contract before any training proposal.',
        matched_control='Exact original train method/real frozen trained encoder and TRAIN tokens; only '
                        'restore the already requested frozen eval mode, hold parameters, token masks and decoder projection.',
        next_action='Execute frozen CPU real-encoder/projection mode and RNG controls; native task impact remains pending.',
        failure_scope='inconclusive',evidence=[plan,ref(path),ref(body),ref(__file__)],
        gate_summary={k:'pending' for k in ('diagnosis','controls','native_backward','resources','independence','constraints')})
    work['candidates'].append(c)
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Registered independent Y59 frozen-language mode diagnosis and preregistered real CPU test.')
finally:
    lock.close()
