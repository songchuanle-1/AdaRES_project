"""Persist an honest near-deadline disposition and Chinese research notes."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor
token = "601bcda714184133920945f8f067b382"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
remaining = (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds()
assert 20 < remaining <= 180, "Only the actual immutable deadline may justify this stop"


def evidence(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


refs = [evidence("pipeline/analyses/" + stem + "_" + token + ".json") for stem in (
    "end_resources_601b", "y46_local_reserve_601b", "y46_cached_native_601b", "y48_launch_601b")]
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y48")
    assert candidate["stage"] == "training"
    candidate["evidence"].append(refs[0])
    work["design_cycle"] = dict(
        stop_reason="tick_deadline_near", attempted_ids=["AdaRES-Y45", "AdaRES-Y46", "AdaRES-Y48"],
        evidence=refs,
        resume_condition="Continue SAME Y46 local independent package with guarded additive upload, actual production/released-init/tools.train/resources and gate review under next owned tick; preserve current Y48 fixed30 and foreign GPU0. Do not claim exhausted routes or fabricate an external block.",
        observed_at_local=datetime.now().astimezone().isoformat(),
        remaining_seconds_at_disposition=remaining)
    path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    notes = dict(actions=[
        "本轮仅AdaRES_project；实际任务归属/closure/最新目标与server2 GPU0/1授权核实，后台调度保持Disabled。begin及monitor/goal仅各一次。快照2026-10-15 10:49:10.174872+08，真实截止11:49:10.174872+08。第0至3段沿用该快照；Y45的preflight运行表是尚未归档的旧状态，不是结束时事实。",
        "Y45自然完成固定30轮，last及scheduler30/best16，原始七源文件与检查点哈希核实、日志归档。Y45/W2/B1各新做两遍9508表达/141场景，FIRST mIoU=.507656778/.510105160/.502214385；相对W2差-.002448383，场景CI[-.009974326,.004838436]，相对B1差+.005442392，CI[-.003483179,.014432856]。所有冻结子组区间跨零、无伤害触发，但不证明无害。训练后成本和重复记录齐全，Y45记complete_inconclusive/failure_scope=inconclusive，不晋升、不调参或否定整个机制。",
        "Y46原冻结TRAIN8真实原生损失/反向和成本已执行，8个新张量均有限非零。成本问题按implementation预注册同ID修订1，仅缓存完全相同选点集合；CPU32真实表达输入行/权重/组顺序精确一致，原生TRAIN8复测通过。热身后候选描述性耗时由3.49–18.66秒降至.627–8.277秒，但仍高于父模型.213–.297秒，不作资源PASS或效果结论。独立七文件本地储备已完成编译/剥离AST父一致性，未上传、未提交、未准入；下一步真实生产released-init/tools.train/资源等门槛具体保留。",
        "Y48同一次原生前向联合捕获：8场景/32表达/128查询，171个非空重复掩码组中31个混合原生标签组，组内字面TP/FP/FN一致。只对最终indicator非空硬掩码等价组传播已有正标签；CPU32、TRAIN8完整原生反向及52个改变查询的固定logits梯度控制通过。实际独立生产包CPU32、released SPFormer真实tools.train两步（新增正标签3）、参数更新、完整模型/AdamW/PolyLR数值恢复、物理父解码器一致和压力测试14.1367GiB通过，新增参数0；六门槛真实审查通过，仅支持可证伪训练。",
        "失败原样保留：本地/远端父文件字节差异守卫、双场景诊断推理尺寸错误、两个已知HF元数据buffer的strict加载错误、Y46构建UTF8错误。源码、日志、诊断检查点、修复前证据均保留。六文件AST/实际远端字节核实、单场景诊断约定、实际两buffer值及完整状态等价校验、显式UTF8分别解决工具问题；没有改变目标机制、模型或容差来制造PASS。",
        "仅提交测试过的Y48七文件：d18c228fd3796b5a87dbe313075b6ee748b2d51b，六门槛/goal/decision哈希收据已绑定，经现有canonical GPU1路径启动。11:39:29–34核实PID8877/配置/工作目录/提交/七文件哈希及原生步数275→284/4686；11:45:52再观测为1/30轮969/4686。GPU0外部PID19723/23970MiB/100%，GPU1 Y48 PID8877/16920MiB/57%，外部进程保留。当前实际训练1个Y48、排队0；预检事实表不冒充更新后状态。",
        "实际产出：新增稳定候选0、Y46有界修订1、Y45冻结完整评估6遍、Y48准入及新训练各1；多个真实CPU/原生/生产/恢复/资源产物及独立Y46本地储备。科学状态Y45 complete_inconclusive、Y46 implementation、Y48 training；本轮执行仍待finish，不提前称COMPLETED。收据用量当前未知，仅本地begin至写收据窗口，排除初始读取、迟到记录和最终回复；缓存属于输入、推理属于输出，不重复相加，不是账单或科学收益。",
    ], decisions=[
        "本轮真实剩余时间已小于180秒，按tick_deadline_near收尾；Y46仍有可执行后续，保留同ID源码、门槛和具体恢复步骤，不宣称路线穷尽或伪造外部阻塞。原截止时间不延长。",
        "Y48固定released SPFormer/seed1999/30轮/128查询/单argmax；诊断更新不部署。终态仍按冻结各9508/141两遍、FIRST、场景聚类及子组伤害/重复/训练后成本比较，不从启动或梯度活动推断效果。决策pipeline/decisions/20261015-113603_next_batch.md。",
        "W2保留科学比较基线；账面Y36的.5110不是科学晋升。Y43/Y44/Y47及历史拒绝证据不重开，不做验证集选择。无其他项目、server6、调度修改、新任务或嵌套后台操作。下次保存的自动化时间未知。",
    ])
    notes_path = ROOT / ("pipeline/reports/notes_" + token + ".json")
    with notes_path.open("x", encoding="utf-8") as stream:
        json.dump(notes, stream, indent=2, ensure_ascii=False)
    print(json.dumps(dict(notes=str(notes_path), remaining_seconds=remaining,
                          stop_reason="tick_deadline_near"), ensure_ascii=False))
finally:
    writer.close()
