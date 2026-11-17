"""Bind the independent package and existing native loader/restoration helpers."""
import ast
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
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
        assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds()>750
        work=tick.load(ROOT/'pipeline/research_work.json')
        candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y54')
        assert candidate['stage']=='implementation'
        for row in candidate['evidence']:
            assert ref(row['path'])==row
        bundle_path='pipeline/analyses/y54_prototype_bundle_'+TOKEN+'.json'
        bundle=tick.load(ROOT/bundle_path)
        for row in bundle['files']:
            assert ref(row['path'])==row
        old_program='pipeline/analyses/y52_native_matched_capture_binding_v2_89ffd8db36cf43ed8cfd75a68c77b4b9.py'
        tree=ast.parse((ROOT/old_program).read_text(encoding='utf-8'))
        payload=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets)))
        payload['files']={Path(row['path']).name:dict(sha256=row['sha256'],base64=base64.b64encode((ROOT/row['path']).read_bytes()).decode('ascii')) for row in bundle['files']}
        payload['parent_normalized']={Path(row['path']).name:row['normalized_sha256'] for row in bundle['source_files']}
        payload['checkpoint_sha256']='e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
        helpers_path='scripts/probe_y52_native_backward_89ff.py'
        helpers=(ROOT/helpers_path).read_text(encoding='utf-8')
        helper_tree=ast.parse(helpers)
        names={'rng_state','restore_rng','cpu_copy','equal_state'}
        selected=[n for n in helper_tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert len(selected)==4
        payload['state_helpers']='\n\n'.join(ast.get_source_segment(helpers,n) for n in selected)
        setup=(ROOT/'scripts/probe_y52_native_matched_capture.py').read_text(encoding='utf-8').split('rows = []\n',1)[0]
        setup=setup.replace('adares_y52_point_target','adares_y54_fine_support')
        marker="for row in PAYLOAD['parent_files']:\n    assert hashlib.sha256((root / row['path']).read_bytes()).hexdigest() == row['sha256']\n"
        assert setup.count(marker)==1
        setup=setup.replace(marker,marker+"    assert hashlib.sha256((root / row['path']).read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest() == PAYLOAD['parent_normalized'][Path(row['path']).name]\n")
        body='scripts/probe_y54_native_gates_fde0.py'
        source='PAYLOAD='+repr(payload)+'\n'+setup+'\n'+(ROOT/body).read_text(encoding='utf-8')
        ast.parse(source,feature_version=(3,8))
        program='pipeline/analyses/y54_native_gates_program_'+TOKEN+'.py'
        plan_path='pipeline/analyses/y54_native_gates_prereg_'+TOKEN+'.json'
        assert not (ROOT/program).exists() and not (ROOT/plan_path).exists()
        plan=dict(complete=True,id='AdaRES-Y54',token=TOKEN,preregistered_at=datetime.now().astimezone().isoformat(),bundle=ref(bundle_path),body=ref(body),helpers=ref(helpers_path),setup_source=ref('scripts/probe_y52_native_matched_capture.py'),
            fixed_test='Independent SAME seven-file prototype, exact raw parent hashes plus normalized local-parent equality, released SPFormer native loader/seed1999. SAME8TRAIN scenes/4batchsize2 pairs: real sparse/FPS/native loss/backward, all6fine-layer finite nonzero valid/attention gradients and zero actual invalid-source gradient; indicator gradient active. Fixed point-largest original pair2warmups/3measures finite reserved<24GiB. Native AdamW/PolyLR2steps with nonempty model/optimizer/scheduler/allRNG exact RAM restore. Matched parameter/state schema. SAME trained-W2 weights parent/candidate8single-scene predict,2warmups/3measures, binary/argmax/IoU parity versus independent repeats plus latency/memory. Read unique existing Y36 best archive hash only, never select by new validation.',
            expected='Corrected invalid fine supports contribute zero gradient while valid native task paths remain finite and active. No added parameters, finite within-card memory and supported single-scene inference behavior within measured repeats; report all limitations or failures honestly, no fixed task gain or tolerance tuning.',
            limits='No checkpoint weights saved or deployed, Git commit, admission, queue or training launch. Fixed trained W2 is diagnostic only. Physically independent namespace; not Y52 or Y50 intervention.',result='pending')
        (ROOT/program).write_text(source,encoding='utf-8',newline='\n')
        plan['program']=ref(program)
        tick.save(ROOT/plan_path,plan)
        for path in (body,'scripts/prepare_y54_native_gates_fde0.py',program,plan_path):
            candidate['evidence'].append(ref(path))
        candidate['stage']='validation'
        candidate['next_action']='Execute independent frozen native gates and supported batchsize1 same-weight predict. Preserve any implementation/diagnostic failure; no automatic Git commit/admission or trained-efficacy claim.'
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(program=ref(program),plan=ref(plan_path)),indent=2))
    finally:
        lock.close()


if __name__=='__main__':
    main()
