"""Preserve an honest closure mismatch without bypassing its check."""
from datetime import datetime
import hashlib
import json
from research_tick_4567 import ROOT, TOKEN, lease, ref, tick, monitor

lease()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    failure_path = 'pipeline/analyses/postflight_failure_4567_' + TOKEN + '.json'
    assert not (ROOT / failure_path).exists()
    checker = __import__('pathlib').Path('C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/closure.py')
    text = checker.read_text(encoding='utf-8')
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if 'Blocked GPU candidate has a queue opening' in line)
    resource_path = 'pipeline/analyses/y52_final_status_v2_4567_' + TOKEN + '.json'
    tick.save(ROOT / failure_path, dict(complete=True, token=TOKEN,
        observed_at_local=datetime.now().astimezone().isoformat(), execution_status='INCOMPLETE',
        finish_exit_code=1, failed_check='CLOSURE_INCOMPLETE: Blocked GPU candidate has a queue opening',
        command_result='POSTFLIGHT FAILED: lease retained; continue this tick or explicitly abort',
        checker=dict(path=str(checker), sha256=hashlib.sha256(checker.read_bytes()).hexdigest(), excerpt='\n'.join(lines[index-9:index+1])),
        live_resource_evidence=ref(resource_path),
        discrepancy='Closure uses own-project active queue occupancy as an opening. Actual GPU0 has foreign PID19723; GPU1 has own Y50 PID31539. No genuinely idle authorized exclusive GPU exists in the saved live observation.',
        disposition='Preserve real blocked Y52. Do not recategorize blocker, fabricate a queue entry or modify checker/goal/context. Abort this tick, not training.',
        abort_status='pending command; verify release separately'))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y52')
    assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
    candidate['evidence'].append(ref(failure_path))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    memory_path = ROOT / 'pipeline/codex_memory.md'
    memory = memory_path.read_text(encoding='utf-8')
    memory = memory.replace('PostflightPENDING, not completed yet.', 'TickINCOMPLETE: finishexit1/Blocked_GPU_candidate_has_a_queue_opening; actualGPU0foreignoccupied/GPU1Y50. Canonicalabortpending; trainingNOTstopped.')
    memory = memory.replace('Current report/notes andseven-sectionmemoryprepared; callfinishonce, trustexplicitPASS/COMPLETE plusabsentownedlease. Ifunfixable, preserveandabort; nopostreleasewrites.', 'CurrentfinishFAILED once: CLOSURE_INCOMPLETE: Blocked GPU candidate has a queue opening. Checkerusesownqueueopeningratherthanforeignliveoccupancy. Failureevidence' + failure_path + '; doNOTrepeatfinish/reclassifyblocker/fakequeue. Canonicalabortnext; verifyreceipt/absentlease; nopostreleasewrites.')
    assert len(memory.splitlines()) < 120
    memory_path.write_text(memory, encoding='utf-8')
    notes_path = ROOT / ('pipeline/analyses/notes_4567_' + TOKEN + '.json')
    notes = tick.load(notes_path)
    notes['actions'][0] = '科学状态：Y50 training；Y52完成CPU准备，真实blocked/exclusive_gpu_slot。执行状态：INCOMPLETE；finish返回1并保留租约，尚待abort释放。'
    notes['actions'].append('真实收尾故障：CLOSURE_INCOMPLETE: Blocked GPU candidate has a queue opening。检查器只按本项目active=1/容量2认定有空位，未反映GPU0外来PID19723的实时占用。保存检查器源码哈希、具体条件及实时占用证据，不改门槛、不造排队项，不重试相同finish。')
    notes['decisions'].append('调用规范abort，仅中止本轮tick并核实释放；保留Y52证据和原有Y50训练，不将本轮称为完成。')
    tick.save(notes_path, notes)
    print(json.dumps(dict(failure=ref(failure_path), stage_unchanged='blocked', next_action='report then abort and verify lease release'), indent=2))
finally:
    lock.close()
