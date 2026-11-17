"""Persist a due resource observation and genuine CPU integration progress."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
import closure
parser=argparse.ArgumentParser()
parser.add_argument('--token',required=True)
parser.add_argument('--resource',required=True)
parser.add_argument('--construction',required=True)
parser.add_argument('--observed-end',required=True)
args=parser.parse_args()


def ref(path):
    resolved=(ROOT/path).resolve()
    assert ROOT in resolved.parents
    return dict(path=path,sha256=hashlib.sha256(resolved.read_bytes()).hexdigest())


owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==args.token and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds()>180
observed=datetime.fromisoformat(args.observed_end)
assert observed.tzinfo and 0 <= (datetime.now(timezone.utc)-observed).total_seconds()<1800
assert args.token in args.resource and args.token in args.construction
resource=tick.load(ROOT/args.resource)
constructed=tick.load(ROOT/args.construction)
assert resource['complete'] and resource['read_only'] and resource['idle_eligible_gpu_ids']==[]
assert constructed['complete'] and constructed['fixed_seed_initial_state_exact'] and constructed['physical_namespace_independent']
assert constructed['additional_parameters']==0 and not constructed['native_forward_executed'] and not constructed['native_backward_executed']
assert resource['Y50']['same_process_confirmed'] and resource['Y50']['native_steps_advanced']
report_path='pipeline/reports/summary_'+args.token+'.md'
report=(ROOT/report_path).read_text(encoding='utf-8')
metric=re.search(r'\[adares_y50_joint_incidence_res\] Best=([0-9.]+) (\d+)/(\d+)',report)
assert metric
best=float(metric.group(1))
view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
prior_path='pipeline/analyses/postflight_failure_4567_4567f5a09ecb4aa89df8f0c2932652df.json'
prior=tick.load(ROOT/prior_path)
checker=Path('C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/closure.py')
assert hashlib.sha256(checker.read_bytes()).hexdigest()==prior['checker']['sha256']
context_path=ROOT/'pipeline/headless/.tick_context.json'
assert hashlib.sha256(context_path.read_bytes()).hexdigest()==owner['context_sha256']
queue=tick.load(context_path)['queue']
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y52')
    assert candidate['stage']=='blocked' and candidate['blocker_kind']=='exclusive_gpu_slot'
    assert datetime.fromisoformat(candidate['retry_after'])<=observed
    for evidence in candidate['evidence']:assert ref(evidence['path'])==evidence
    bundle_path='pipeline/analyses/y52_prototype_bundle_4567_4567f5a09ecb4aa89df8f0c2932652df.json'
    for row in tick.load(ROOT/bundle_path)['files']:assert ref(row['path'])==row
    rejection=closure.exclusive_gpu_slot_available(candidate,queue['servers'],queue['active_tasks'])
    assert rejection is True, 'Known closure condition changed: reassess instead of aborting'
    next_retry=(observed+timedelta(hours=1)).isoformat()
    previous_retry=candidate['retry_after']
    evidence_paths=[args.resource,args.construction,'scripts/probe_adares_due_resource_status.py',
        'scripts/probe_y52_full_cpu_construction.py','scripts/prepare_y52_cpu_construction.py',
        'pipeline/analyses/y52_cpu_construction_prereg_'+args.token+'.json',
        'pipeline/analyses/y52_cpu_construction_program_'+args.token+'.py']
    seen={row['path'] for row in candidate['evidence']}
    candidate['evidence'].extend(ref(path) for path in evidence_paths if path not in seen)
    candidate.update(retry_after=next_retry,last_resource_check_at=observed.isoformat(),
        blocker='Due live observation: GPU0 foreign PID19723, GPU1 same Y50 PID31539; no eligible idle authorized slot. CPU full-model imports/construction/initial-state control now also completed. Actual sparse TRAIN matching/backward/resources still require an exclusive GPU.',
        next_action='Resume SAME frozen Y52 with actual native sparse TRAIN matching/forward/backward, shared-query gradients, resource and full restore controls only after fresh idle authorization checks and exclusive GPU lease. CPU construction is NOT native gate PASS. No automatic Y52 Git commit authorization.')
    candidate['gate_summary']['independence']='physical namespace plus full CPU construction/fixed-seed initial state tested; native path remains pending'
    candidate['attempted_alternatives'].append('New full independent IPDN CPU construction at seed1999 matches actual W2 all1074 state tensors exactly; zero added parameters, no native forward/backward. Due resource observation still shows both authorized slots occupied.')
    audit_path='pipeline/analyses/y52_due_closeout_'+args.token+'.json'
    assert not (ROOT/audit_path).exists()
    tick.save(ROOT/audit_path,dict(complete=True,token=args.token,observed_at_local=datetime.now().astimezone().isoformat(),
        execution_status='INCOMPLETE: canonical abort pending',science='Y50 live training; SAME Y52 CPU construction passed, exclusive-GPU blocked',
        resource=ref(args.resource),construction=ref(args.construction),prototype=ref(bundle_path),
        previous_retry_after=previous_retry,next_retry_after=next_retry,
        checker_unchanged=prior['checker']['sha256'],known_queue_opening_rejecting_predicate=True,
        finish_invoked=False,native_forward_or_backward=False,new_candidates=0,new_CPU_integration_tests=1,
        model_changes=0,admissions=0,launches=0,
        limitation='Fixed-seed full CPU initialization/control only, not native TRAIN gate, accuracy benefit, novelty, admission or resource Pareto evidence.'))
    candidate['evidence'].append(ref(audit_path))
    work['design_cycle']=dict(stop_reason='external_resource_blocked',attempted_ids=['AdaRES-Y52'],
        evidence=[ref(args.resource),ref(args.construction),ref(audit_path)],
        resume_condition=candidate['unblock_condition'],retry_after=next_retry)
    tick.save(ROOT/'pipeline/research_work.json',work)
    before,after=resource['Y50']['progress_before'],resource['Y50']['progress_after']
    memory_path=ROOT/'pipeline/codex_memory.md'
    lines=memory_path.read_text(encoding='utf-8').splitlines()
    replacements={
        '- 当前token':f'- 当前token：{args.token}；快照{view["generated_at"]}；不可改deadline{view["deadline_at"]}；design/续做同一Y52。',
        '- 本轮INCOMPLETE':'- 本轮INCOMPLETE：到期资源核验仍无空闲授权卡，已知收尾谓词未改变；保存真实新CPU结果后规范abort，尚待核实释放，训练未停止。',
        '- 本轮begin监控':f'- 本轮begin监控Y50 best {best:.4f}、{metric.group(2)}/{metric.group(3)}；序列.3115/.3657/.4323/.4456。当前stdout约mIoU .446/Acc50 .490/Acc25 .557/spmIoU .411，是早期验证，不是项目最佳或已证明收益。',
        '- Y52仍blocked':f'- Y52仍blocked/exclusive_gpu_slot/server-2；真实到期核验后retry更新为{next_retry}，仅计时器不是释放证据。原型字节不变、无实现修订/提交/部署。下一步仍为原生匹配/反向与资源恢复验收。',
        '- 本轮begin16':f'- 本轮begin17:49:40+08：active1/running1/claim0/wait0/target2/准入预算1/储备1of2/准备预算1；新CPU集成测试1，模型改动0、新候选0、准入0、启动0。',
        '- 本轮实时GPU占用':'- 本轮真实客户端17:52:51.568875至17:52:56.087580+08复核：GPU0外来PID19723/92%/23664MiB；GPU1本项目Y50 PID31539/81%/22700MiB；无空闲授权卡，未预约或取得GPU诊断租约。',
        '- Y50原log':f'- Y50原log train_adares_y50_joint_incidence_res_20261016_145628.log；本轮两次独立读数Train[{after[0]}/{after[1]}]步{before[2]}→{after[2]}/{after[3]}，已确认实际进度，无有限尾部崩溃标记。released SPFormer/seed1999/fixed30/128query/singleargmax不变。',
        '- 上轮eaa6':'- 上轮44f779于16:55:34规范abort，收据INCOMPLETE；16:56:48锁不存在且Y50仍在。旧pending是释放前措辞，绝不重开旧token。',
        '- 本轮唯一begin':'- 本轮唯一begin --auto-steady后先读compact/按ID续Y52。到期才重查占用；新增完整独立模型CPU构造与固定种子初态对照，不重复旧loss-only控制，不为储备凑候选。',
        '- 本轮核对证据':f'- 本轮证据{audit_path}；六段报告留档，简报按visible_reporting.md；按已知不可通过谓词不重放finish，规范abort后仅只读核实，释放后不写项目。'
    }
    for prefix,value in replacements.items():
        indexes=[i for i,line in enumerate(lines) if line.startswith(prefix)]
        assert len(indexes)==1,prefix
        lines[indexes[0]]=value
    position=next(i for i,line in enumerate(lines) if line.startswith('- 全模型backbone'))
    lines.insert(position,'- 新CPU集成检查17:56完成：实际独立IPDN及W2各164629637参数、39984005可训练参数、896参数张量/1074状态张量；固定seed1999初始状态逐字节完全相同。只构造/导入，无GPU/前后向/权重保存，不代替原生门禁。')
    assert len(lines)<120
    memory_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    notes_path='pipeline/analyses/notes_'+args.token+'.json'
    tick.save(ROOT/notes_path,dict(actions=[
        '执行状态INCOMPLETE，待规范abort核实释放；科学状态Y50训练已确认、同一Y52完成新的CPU集成控制但仍等待独占GPU。目标、身份和Disabled调度已核对，本轮未改目标、门禁、调度或其他项目。',
        f'客户端17:52:51.57至17:52:56.09+08到期只读复查：GPU0外来PID19723/92%/23664MiB，GPU1本项目Y50 PID31539/81%/22700MiB；没有空闲授权卡，未取得或预留GPU诊断租约。新回查条件时间{next_retry}，不是调度承诺或空闲证明。',
        f'Y50同一进程两次独立原生步数{before[2]}→{after[2]}/{after[3]}，第{after[0]}/{after[1]}轮，实际训练进度已确认。begin最佳mIoU {best:.4f}；本轮stdout约mIoU .446/Acc@.5 .490/Acc@.25 .557/spmIoU .411。仅早期结果，尚不如历史基线，不是项目最佳；延迟和峰值显存未知。',
        '先冻结测试范围及七文件字节，随后在CUDA隐藏的CPU环境实际导入完整独立Y52与W2，固定seed1999构造。双方参数164629637、可训练39984005、状态1074项，初始状态逐字节完全相同，新增参数0；说明包级导入/构造/初始化一致，不证明原生匹配、反向、资源、恢复或训练收益。',
        '原型无改动、无新候选或实现修订、无权重保存/提交/部署、准入0/新启动0。GPU占用只阻断真正需要独占卡的下一项原生测试；本轮补完尚未做过的完整CPU构造检查，旧loss-only控制不重跑。',
        '完整报告保留六段和原begin表；程序只读核验已知queue-opening拒绝谓词仍为真且checker SHA未变，不重放必失败finish，不改阻塞类型或造队列绕过。收据将保留实际用量/产出，当前尚未读取；缓存/推理不重复相加、不当账单，释放后不补写报告。'],
        decisions=['保留同一Y52原机制和冻结实现，依据真实新资源证据延后回查；只在授权卡确实空闲且独占租约取得后，续做原生匹配/反向与资源/恢复测试，六门禁通过前不准入。',
            'CPU构造对照通过只是集成准备，不宣传模型有效或轻量Pareto；Y50继续固定30训练，不移动/重启或调验证参数。',
            '同一收尾不一致已留证且未改变，按未完成保存并abort，不反复向用户推送已知故障。实际下一自动化时间未知，不新增审批或更改通知策略。']))
    print(json.dumps(dict(token=args.token,notes=str(ROOT/notes_path),evidence=ref(audit_path),
        stage=candidate['stage'],next_retry_after=next_retry,new_CPU_integration_tests=1,new_candidates=0,admissions=0,launches=0,
        next_action='canonical report then abort and verify release'),ensure_ascii=False,indent=2))
finally:lock.close()
