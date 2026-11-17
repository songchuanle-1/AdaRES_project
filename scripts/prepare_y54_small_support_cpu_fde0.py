"""Freeze edge controls after actual native gates; never alter the tested bundle."""
import ast
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
from register_y53_padding_diagnosis_fde0 import ref,TOKEN
import foreground_tick as tick
import pipeline_monitor as monitor


def main():
    lock=monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token']==TOKEN and owner['thread_id']==os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds()>300
        work=tick.load(ROOT/'pipeline/research_work.json')
        candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y54')
        assert candidate['stage']=='validation'
        for row in candidate['evidence']:
            assert ref(row['path'])==row
        gates_path='pipeline/analyses/y54_native_gates_fde0_'+TOKEN+'.json'
        gates=tick.load(ROOT/gates_path)
        assert gates['complete'] and all(gates['nonempty_optimizer_restore'].values())
        original='pipeline/analyses/y54_native_gates_program_'+TOKEN+'.py'
        tree=ast.parse((ROOT/original).read_text(encoding='utf-8'))
        payload=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets)))
        body='scripts/probe_y54_small_support_cpu_fde0.py'
        source='PAYLOAD='+repr(payload)+'\n'+(ROOT/body).read_text(encoding='utf-8')
        ast.parse(source,feature_version=(3,8))
        program='pipeline/analyses/y54_small_support_cpu_program_'+TOKEN+'.py'
        plan_path='pipeline/analyses/y54_small_support_cpu_prereg_'+TOKEN+'.json'
        assert not (ROOT/program).exists() and not (ROOT/plan_path).exists()
        plan=dict(complete=True,id='AdaRES-Y54',token=TOKEN,preregistered_at=datetime.now().astimezone().isoformat(),native_gates=ref(gates_path),body=ref(body),
            cases='Exactly five adversarial component fixtures: valid3D supports[1,512],[8,512],[128,512],[406,512],[512,512]. Preserve native512fine/128queries/256fixed fixture seed indices/6layers and fixed seed1999. No real FPS claimed. Exact isolated decoder; every corrected neighborhood nonempty, all outputs/backward finite and invalid-source gradients0; all-valid exact parent same-state indicator control.',
            prediction='Validity propagates even when padded supports outnumber native neighborhood size, without all-masked attention. Full-valid support should leave original forward unchanged. Failure is implementation evidence, never tuned away.',
            limits='Component edge controls only, not a new candidate/model revision or trained task result; same immutable seven-file bundle, no weights saved/commit/admission.',result='pending')
        (ROOT/program).write_text(source,encoding='utf-8',newline='\n')
        plan['program']=ref(program)
        tick.save(ROOT/plan_path,plan)
        for path in (gates_path,body,'scripts/prepare_y54_small_support_cpu_fde0.py',program,plan_path):
            if path not in {row['path'] for row in candidate['evidence']}:
                candidate['evidence'].append(ref(path))
        candidate['next_action']='Execute fixed adversarial small-support/all-valid exact native component controls, then truthful bounded gate and scoped commit-permission review. No model change or training.'
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(program=ref(program),plan=ref(plan_path)),indent=2))
    finally:
        lock.close()


if __name__=='__main__':
    main()
