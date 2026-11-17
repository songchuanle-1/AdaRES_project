"""Freeze a meaningful trained-parent inference readout without changing models."""
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
from prepare_y52_native_backward_89ff import ref, TOKEN
import foreground_tick as tick
import pipeline_monitor as monitor


def main():
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 420
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y52')
        for row in candidate['evidence']:
            assert ref(row['path']) == row
        old = 'pipeline/analyses/y52_resources_batch_compat_program_' + TOKEN + '.py'
        prior = 'pipeline/analyses/y52_resources_batch_compat_89ff_' + TOKEN + '.json'
        result = tick.load(ROOT / prior)
        assert result['complete'] and all(r['finite_backward'] for r in result['stress'])
        source = (ROOT / old).read_text(encoding='utf-8')
        marker = 'candidate = model.cpu()\n'
        replacement = '''candidate = model.cpu()
import hashlib
checkpoint_path = Path('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth')
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
weights = dict(checkpoint['model'])
extras = set(weights) - set(candidate.state_dict())
assert extras.issubset({'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'})
buffers = dict(candidate.named_buffers())
for key in extras:
    assert key in buffers and not buffers[key].requires_grad
    assert torch.equal(weights[key].cpu(), buffers[key].cpu())
    del weights[key]
candidate.load_state_dict(weights, strict=True)
del weights, checkpoint
'''
        assert source.count(marker) == 1
        source = source.replace(marker, replacement)
        marker = "queries=[int(torch.as_tensor(v).numel()) for v in output['indi']])"
        replacement = "queries=[int(torch.as_tensor(v).numel()) for v in output['indi']], confidence=[torch.as_tensor(v).detach().cpu().clone() for v in output['indi']], selected=[int(torch.as_tensor(v).argmax().item()) for v in output['indi']])"
        assert source.count(marker) == 1
        source = source.replace(marker, replacement)
        marker = "return dict(mask_points_changed=sum(int((a != b).sum().item()) for a, b in zip(left['masks'], right['masks'])),"
        replacement = "return dict(probability_max_abs=max(float((a-b).abs().max().item()) for a,b in zip(left['masks'],right['masks'])), binary_mask_points_changed=sum(int(((a>0.5)!=(b>0.5)).sum().item()) for a,b in zip(left['masks'],right['masks'])), selected_query_changes=sum(a!=b for a,b in zip(left['selected'],right['selected'])), confidence_max_abs=max(float((a-b).abs().max().item()) for a,b in zip(left['confidence'],right['confidence'])), mask_points_changed=sum(int((a != b).sum().item()) for a, b in zip(left['masks'], right['masks'])),"
        assert source.count(marker) == 1
        source = source.replace(marker, replacement)
        marker = 'result = dict(complete=True, id=\'AdaRES-Y52\', exact_initial_state_equal=True,'
        replacement = "result = dict(complete=True, id='AdaRES-Y52', diagnostic_state='fixed_trained_W2_not_trained_Y52', native_mask_threshold=0.5, exact_initial_state_equal=True,"
        assert source.count(marker) == 1
        source = source.replace(marker, replacement)
        ast.parse(source, feature_version=(3, 8))
        program = 'pipeline/analyses/y52_predict_readout_program_' + TOKEN + '.py'
        plan_path = 'pipeline/analyses/y52_predict_readout_prereg_' + TOKEN + '.json'
        assert not (ROOT / program).exists() and not (ROOT / plan_path).exists()
        plan = dict(complete=True, id='AdaRES-Y52', token=TOKEN, preregistered_at=datetime.now().astimezone().isoformat(), prior=ref(prior), original_program=ref(old), fixed_input='Same8TRAIN scenes/32expressions and native single-scene predict. Fixed W2 trained checkpoint already used prospectively for gradient diagnosis; strict real-state loading with exact legacy-static buffers only.', readout='Retain raw probability inequality counts; additionally measure max probability/confidence difference, native threshold0.5 binary-mask differences, selected argmax changes and per-expression IoU differences, EACH versus independent same-model repeats. No tolerance tuning or accuracy claim.', prediction='Inference unchanged by point-target training correction; compare matched output differences with repeat differences. Unexplained behavioral discrepancies remain inconclusive, never forced PASS.', limits='No new model weights, model revision, validation selection, commit, admission or training. Fixed W2 checkpoint is NOT a trained Y52 checkpoint.', result='pending')
        (ROOT / program).write_text(source, encoding='utf-8', newline='\n')
        plan['program'] = ref(program)
        tick.save(ROOT / plan_path, plan)
        for path in (prior, 'scripts/prepare_y52_predict_readout_89ff.py', program, plan_path):
            if path not in {e['path'] for e in candidate['evidence']}:
                candidate['evidence'].append(ref(path))
        candidate['next_action'] = 'Run fixed-trained-parent native inference readout with independent repeats and the original native binary/argmax decision. Probability inequality alone is not behavioral change; no forced parity PASS or automatic commit.'
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(program=ref(program), plan=ref(plan_path)), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
