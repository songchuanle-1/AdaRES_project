"""Bind a completed real TRAIN collate control without promoting native gates."""
import argparse
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


def ref(path):
    resolved = (ROOT / path).resolve()
    assert ROOT in resolved.parents and resolved.is_file(), path
    return dict(path=path, sha256=hashlib.sha256(resolved.read_bytes()).hexdigest())


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--token', required=True)
parser.add_argument('--result', required=True)
parser.add_argument('--result-sha256', required=True)
parser.add_argument('--program-sha256', required=True)
args = parser.parse_args()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
    assert owner['token'] == args.token and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
    assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 180
    assert args.token in args.result and ref(args.result)['sha256'] == args.result_sha256
    result = tick.load(ROOT / args.result)
    assert result['complete'] and result['no_GPU']
    assert result['actual_production_TRAIN_dataset_collate_executed']
    assert result['frozen_TRAIN_scenes'] == 8 and result['native_two_scene_batches'] == 4 and result['expressions'] == 32
    assert result['single_vs_two_scene_counts_exact'] and result['expanded_point_quality_error'] == 0.
    for key in ('candidate_model_constructed', 'native_full_model_forward_executed', 'native_full_model_backward_executed', 'model_weights_saved'):
        assert result[key] is False
    program_path = 'pipeline/analyses/y52_train_collate_program_' + args.token + '.py'
    prereg_path = 'pipeline/analyses/y52_train_collate_prereg_' + args.token + '.json'
    assert ref(program_path)['sha256'] == args.program_sha256
    prereg = tick.load(ROOT / prereg_path)
    assert prereg['token'] == args.token and ref(prereg['body']['path']) == prereg['body']
    assert ref(prereg['prototype']['path']) == prereg['prototype']
    resource_audit_path = 'pipeline/analyses/y52_resource_recheck_' + args.token + '.json'
    resource_audit = tick.load(ROOT / resource_audit_path)
    assert resource_audit['token'] == args.token and not resource_audit['finish_invoked']
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y52')
    assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
    for row in candidate['evidence']:
        assert ref(row['path']) == row
    notes_path = ROOT / ('pipeline/analyses/notes_' + args.token + '.json')
    notes = tick.load(notes_path)
    assert len(notes['actions']) == 6 and '新CPU测试0' in notes['actions'][3]
    action = (
        f"本轮新增真实TRAIN双场景CPU数据流检查1项：沿用冻结8场景/32表达组成4个生产batch_size2批次，"
        f"与独立单场景collate逐表达比对计数和注释完全一致；含{result['mixed_expression_superpoints']}个混标签表达-超点实例，"
        "空/满/奇偶三种固定mask的精确点IoU误差0。只排除这组真实多场景偏移/标签计数接口错误，不是学习预测或原生匹配频率，更不是训练收益。"
    )
    notes['actions'][3] = action + '旧CPU构造和组件对照未重跑；原型七文件无改动、无实现修订、新候选0、准入0、新启动0。'
    notes['actions'][5] = notes['actions'][5].replace('实际产出为本次资源证据与真实阻塞/收尾记录', '实际产出为1项真实双场景CPU集成结果、资源证据与真实阻塞/收尾记录')
    notes['decisions'].append('双场景CPU数据流结果通过但不提升为原生前后向或资源门禁PASS；仍保留同一Y52及其全部旧负证据，空闲独占GPU后续做真实稀疏模型匹配/梯度与恢复。')
    memory_path = ROOT / 'pipeline/codex_memory.md'
    lines = memory_path.read_text(encoding='utf-8').splitlines()
    queue_lines = [i for i, line in enumerate(lines) if line.startswith('- 本轮队列')]
    begin_lines = [i for i, line in enumerate(lines) if line.startswith('- 本轮唯一begin')]
    native_lines = [i for i, line in enumerate(lines) if line.startswith('- 全模型backbone')]
    assert len(queue_lines) == len(begin_lines) == len(native_lines) == 1
    assert '本轮新CPU测试0' in lines[queue_lines[0]]
    lines[queue_lines[0]] = lines[queue_lines[0]].replace('本轮新CPU测试0', '本轮新CPU数据流集成测试1')
    lines[begin_lines[0]] = '- 本轮唯一begin --auto-steady后先读compact/按ID续Y52。新增真实双场景TRAIN collate CPU检查1项，到期只查资源一次；旧CPU构造/组件不重跑，无占位候选。'
    lines.insert(native_lines[0], '- 新双场景TRAIN检查19:55完成：冻结8场景/32表达、4个生产batch_size2批次，跨场景计数/注释与独立单场景精确一致；5个混标签表达-超点实例，3类固定mask的展开点IoU误差0。没有模型前后向或GPU门禁PASS。')
    assert len(lines) < 120 and sum(line.startswith('## ') for line in lines) == 7
    audit_path = 'pipeline/analyses/y52_train_collate_closeout_' + args.token + '.json'
    assert not (ROOT / audit_path).exists()
    paths = [args.result, program_path, prereg_path, 'scripts/probe_y52_train_collate_cpu.py',
        'scripts/prepare_y52_train_collate_cpu.py', 'scripts/record_y52_train_collate_cpu.py']
    references = [ref(path) for path in paths]
    audit = dict(complete=True, token=args.token, recorded_at_local=datetime.now().astimezone().isoformat(),
        execution_status='INCOMPLETE: canonical abort/release check pending',
        science='New real batchsize2 CPU dataflow control passed; SAME Y52 still GPU-blocked, native gates unrun',
        references=references, resource_disposition=ref(resource_audit_path),
        new_CPU_dataflow_integration_tests=1, model_changes=0, implementation_revisions=0,
        admissions=0, launches=0, new_candidates=0, native_full_model_backward=False,
        limitation=result['limitation'])
    # All inputs and existing memory anchors are validated before mutation.
    tick.save(ROOT / audit_path, audit)
    seen = {row['path'] for row in candidate['evidence']}
    candidate['evidence'].extend(row for row in references + [ref(audit_path)] if row['path'] not in seen)
    candidate['attempted_alternatives'].append('New actual production TRAIN batchsize2 CPU collate integration: same frozen8 scenes/32 expressions,4 real paired batches, single-vs-merged counts/annotations exact including5 mixed-label expression-superpoints; no full model/matching/backward/GPU-resource test.')
    candidate['gate_summary']['controls'] = 'CPU_components_and_real_TRAIN_batchsize2_dataflow_only; native_model_gates_pending; v1_failure_retained'
    work['design_cycle']['evidence'].append(ref(audit_path))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    tick.save(notes_path, notes)
    memory_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(token=args.token, evidence=ref(audit_path), candidate_stage=candidate['stage'],
        new_CPU_dataflow_integration_tests=1, native_full_model_backward=False,
        admissions=0, launches=0, memory_lines=len(lines)), ensure_ascii=False, indent=2))
finally:
    lock.close()
