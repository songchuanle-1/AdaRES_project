"""Preserve the failed test; strictly verify legacy static BERT buffer extras."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
TOKEN = '89ffd8db36cf43ed8cfd75a68c77b4b9'


def ref(relative):
    path = (ROOT / relative).resolve()
    assert ROOT in path.parents and path.is_file()
    return dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 900
        for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
            assert ref('pipeline/headless/' + name)['sha256'] == owner[key]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y52')
        assert candidate['stage'] == 'validation'
        for row in candidate['evidence']:
            assert ref(row['path']) == row
        old_program = 'pipeline/analyses/y52_native_backward_program_' + TOKEN + '.py'
        old_plan = 'pipeline/analyses/y52_native_backward_prereg_' + TOKEN + '.json'
        failure_path = 'pipeline/analyses/y52_native_backward_89ff_' + TOKEN + '.log'
        failure = (ROOT / failure_path).read_text(encoding='utf-8')
        assert 'Unexpected key(s) in state_dict' in failure
        assert 'text_encoder.embeddings.position_ids' in failure and 'text_encoder.embeddings.token_type_ids' in failure
        source = (ROOT / old_program).read_text(encoding='utf-8')
        old = "            model.load_state_dict(checkpoint['model'], strict=True)\n"
        assert source.count(old) == 1
        replacement = '''            weights = dict(checkpoint['model'])
            allowed_static = {'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'}
            legacy_static_buffers_verified = sorted(set(weights) - set(model.state_dict()))
            assert set(legacy_static_buffers_verified).issubset(allowed_static)
            runtime_buffers = dict(model.named_buffers())
            for key in legacy_static_buffers_verified:
                assert key in runtime_buffers and key not in model.state_dict()
                assert torch.equal(weights[key].detach().cpu(), runtime_buffers[key].detach().cpu())
                assert not runtime_buffers[key].requires_grad
                del weights[key]
            model.load_state_dict(weights, strict=True)
            del weights
'''
        source = source.replace(old, replacement)
        marker = '            del loss, logs, corrected, parent, repeated, scores, live_gradients, batch\n'
        assert source.count(marker) == 1
        source = source.replace(marker, "            print(json.dumps({'paired_partial': paired_rows[-1]}), flush=True)\n" + marker)
        marker = '    paired_same_forward=True, parent_criterion_byte_bound=True, candidate_model_revisions=0,\n'
        assert source.count(marker) == 1
        source = source.replace(marker, marker + '    legacy_static_buffers_verified=legacy_static_buffers_verified,\n')
        ast.parse(source, feature_version=(3, 8))
        program_path = 'pipeline/analyses/y52_native_backward_buffer_compat_program_' + TOKEN + '.py'
        plan_path = 'pipeline/analyses/y52_native_backward_buffer_compat_prereg_' + TOKEN + '.json'
        assert not (ROOT / program_path).exists() and not (ROOT / plan_path).exists()
        plan = dict(complete=True, id='AdaRES-Y52', token=TOKEN,
            preregistered_at=datetime.now().astimezone().isoformat(),
            original_plan=ref(old_plan), original_program=ref(old_program), preserved_failure=ref(failure_path),
            failure_scope='implementation: fixed-checkpoint serialization compatibility, not model mechanism',
            change='Only permit the two named legacy nonpersistent BERT buffers after exact equality to current runtime buffers, then retain strict=True for every actual model state entry. Add per-batch partial scalar/gradient summaries to the raw log to preserve evidence on later failures.',
            prediction='If the legacy keys are exactly regenerated static buffers, strict real-parameter loading succeeds and the unchanged paired native/optimizer/restore protocol can finish. Any different or unknown state remains a failure.',
            fixed_protocol='All original samples, states, native matches, repeats, optimizer settings and complete-restore checks unchanged. No model/config changes, checkpoint writes, threshold selection, commit or admission.',
            candidate_model_revisions=0, result='pending')
        (ROOT / program_path).write_text(source, encoding='utf-8', newline='\n')
        plan['program'] = ref(program_path)
        tick.save(ROOT / plan_path, plan)
        for path in (failure_path, 'scripts/prepare_y52_backward_buffer_compat_89ff.py', program_path, plan_path):
            candidate['evidence'].append(ref(path))
        candidate['diagnostic_revision_history'].append(dict(scope='checkpoint_loader_not_model',
            preregistered_at=plan['preregistered_at'], change=plan['change'], prediction=plan['prediction'],
            failure_evidence=[ref(failure_path)], evidence=[ref(plan_path)], result='pending'))
        candidate['next_action'] = 'Execute SAME paired native backward/complete restore protocol with strictly verified legacy-static-buffer checkpoint compatibility. Preserve original failure; no candidate model revision or admission.'
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(program=ref(program_path), plan=ref(plan_path),
            preserved_failure=ref(failure_path), model_revisions=0, strict_real_parameter_load=True), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
