"""Preserve failed evidence; split native eval with a None-returning train override."""
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
from register_y53_padding_diagnosis_fde0 import ref, TOKEN
import foreground_tick as tick
import pipeline_monitor as monitor


def main():
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds() > 400
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y53')
        for row in candidate['evidence']:
            assert ref(row['path']) == row
        original = 'pipeline/analyses/y53_padding_native_program_' + TOKEN + '.py'
        failure = 'pipeline/analyses/y53_padding_native_fde0_' + TOKEN + '.log'
        text = (ROOT / failure).read_text(encoding='utf-8')
        assert "AttributeError: 'NoneType' object has no attribute 'state_dict'" in text
        source = (ROOT / original).read_text(encoding='utf-8')
        before = 'model = native.IPDN(**cfg.model).cuda().eval()'
        assert source.count(before) == 1
        source = source.replace(before, 'model = native.IPDN(**cfg.model).cuda()\nmodel.eval()')
        ast.parse(source, feature_version=(3, 8))
        program = 'pipeline/analyses/y53_padding_native_eval_compat_program_' + TOKEN + '.py'
        plan_path = 'pipeline/analyses/y53_padding_native_eval_compat_prereg_' + TOKEN + '.json'
        assert not (ROOT / program).exists() and not (ROOT / plan_path).exists()
        plan = dict(complete=True, id='AdaRES-Y53', token=TOKEN, preregistered_at=datetime.now().astimezone().isoformat(),
            original_program=ref(original), failure_evidence=ref(failure),
            failure_scope='diagnostic invocation compatibility, not model or hypothesis',
            change='Only split model construction/CUDA assignment from model.eval(). Native IPDN.train override does not return self; do not assign eval result. Every source binding, checkpoint, sample, control, gradient and assertion remains unchanged.',
            prediction='Model remains the actual native IPDN instance in eval mode, so the unchanged fixed TRAIN protocol can run. Other errors or leakage observations remain visible, never forced PASS.',
            candidate_model_revisions=0, result='pending')
        (ROOT / program).write_text(source, encoding='utf-8', newline='\n')
        plan['program'] = ref(program)
        tick.save(ROOT / plan_path, plan)
        for path in (failure, 'scripts/repair_y53_eval_return_fde0.py', program, plan_path):
            candidate['evidence'].append(ref(path))
        candidate.setdefault('diagnostic_revision_history', []).append(dict(scope='eval_return_invocation_not_model', preregistered_at=plan['preregistered_at'], change=plan['change'], prediction=plan['prediction'], failure_evidence=[ref(failure)], evidence=[ref(plan_path)], result='pending'))
        candidate['next_action'] = 'Execute unchanged native TRAIN padding/repeat/valid protocol with only split eval assignment compatibility; original failure retained, no model change or gate PASS.'
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(program=ref(program), plan=ref(plan_path)), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
