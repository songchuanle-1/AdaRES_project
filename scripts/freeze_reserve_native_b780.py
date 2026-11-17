"""Prepare two source-bound, fail-closed native probes without running CUDA."""

import ast
from datetime import datetime
from research_b780 import ROOT,TOKEN,lease,ref,save,bind,tick,monitor
lease()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    old=ROOT/'pipeline/analyses/y56_native_batch2_222ed2a6e80c43f6b1de2d050be97fa0_program.py'
    original=old.read_text(encoding='utf-8')
    tree=ast.parse(original)
    frozen=ast.literal_eval(tree.body[0].value)
    frozen['commit']='05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea'
    start=tree.body[1].lineno
    stop=next(n.lineno for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='records' for t in n.targets))
    setup=''.join(original.splitlines(keepends=True)[start-1:stop-1])
    dataset=tick.load(ROOT/('pipeline/analyses/y58_data_'+TOKEN+'.json'))['dataset_source']
    for number in (58,59):
        c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y'+str(number))
        body=ROOT/('scripts/probe_y'+str(number)+'_native_b780.py')
        program='FROZEN='+repr(frozen)+'\nSCOPE='+repr(dict(dataset_source=dataset))+'\n'+setup+'\n'+body.read_text(encoding='utf-8')
        ast.parse(program,feature_version=(3,8))
        compiled=compile(program,'<pending-native>','exec')
        guarded=False
        try:
            exec(compiled,dict(P=dict(physical_gpu=None)))
        except AssertionError:
            guarded=True
        assert guarded,'Missing authorized physical GPU must fail before imports/model/CUDA work'
        path='pipeline/analyses/y'+str(number)+'_native_'+TOKEN+'_program.py'
        assert not (ROOT/path).exists()
        (ROOT/path).write_text(program,encoding='utf-8',newline='\n')
        plan=save(path.replace('_program.py','_prereg.json'),dict(complete=True,id=c['id'],
            preregistered_at=datetime.now().astimezone().isoformat(),program=ref(path),body=ref(body),
            reused_readonly_setup=ref(old),dataset_source=dataset,
            protocol=('Fixed focal0/mate1/mate2/repeat, nativeBNtrain vs fixedrunningstats, dropout disabled both; '
                      'canonical voxel coordinates, upstream activation and slot-confidence readouts; do not infer harm from combined losses.'
                      if number==58 else 'Exact same native batch0/1 and seed1999, original/fixedfrozen-mode/repeats; '
                      'hold all other dropout off and BN training, actual loss/backward, preserve every signed delta.'),
            test_scope='Trained-parent diagnosis only, not candidate admission; no optimizer or saved weights.',
            source_commit=frozen['commit'],physical_GPU_authorization_required=[0,1],
            before_any_execution='Use canonical runner with NEW current foreground token, fresh idle/GPU process checks and exclusive lease. '
                                 'Rebind changed source only after exact immutable source/checkpoint/data validation.',
            Python38_parse_compile=True,missing_GPU_entry_guard_passed=True,
            native_executed=False,result='pending'))
        validation=save('pipeline/analyses/y'+str(number)+'_native_static_'+TOKEN+'.json',dict(
            complete=True,id=c['id'],program=ref(path),parse_compile_Python38=True,
            missing_GPU_refused_before_CUDA=True,native_executed=False,
            observed_at=datetime.now().astimezone().isoformat(),
            limitation='Static/entrypoint checks, not actual native execution or a scientific PASS.'))
        bind(c,ref(path),plan,validation,ref(body),ref(__file__))
        for stem in (['y58_cpu','y58_data'] if number==58 else ['y59_cpu']):
            bind(c,ref('pipeline/analyses/'+stem+'_'+TOKEN+'.json'),ref('pipeline/analyses/'+stem+'_'+TOKEN+'.log'))
        c['next_action']=('When an authorized exclusive GPU is genuinely free, run SAME frozen native diagnostic '
                          'with current lease/source checks, interpret task relevance and preserve negative/repeat evidence; '
                          'no candidate model, admission or new training yet.')
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Prepared two immutable native diagnostics; compile/entry guards passed, CUDA tests not executed.')
finally:
    lock.close()
