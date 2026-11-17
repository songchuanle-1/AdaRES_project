"""Admit and dispatch ONLY explicitly authorized Y50 through existing gates."""
import argparse
from datetime import datetime,timezone
import hashlib,json,math,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import closure
import foreground_tick as tick
import pipeline_monitor as monitor
TOKEN='75486f9263d34214b3bd2e1412c69639'
OLD='1cb15474bfd346dcbc34837f3e9b79d3'
COMMIT='26ee1cec57c53ee2f94d6969d0bb637565b6ed59'
CONFIG='configs_song/model_paper5/adares_y50_joint_incidence_res.yaml'
parser=argparse.ArgumentParser()
parser.add_argument('operation',choices=('admit','dispatch'))
args=parser.parse_args()
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==TOKEN and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds()>240
def ref(path):
    p=(ROOT/path).resolve()
    assert ROOT in p.parents and p.is_file()
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def prior(stem):return 'pipeline/analyses/'+stem+'_'+OLD+'.json'
def read(path):
    r=tick.load(ROOT/path)
    assert r['complete']
    return r
def check_commit():
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==COMMIT
    bundle=read(prior('y50_production_bundle_1cb1'))
    for row in bundle['files']:
        assert ref(row['path'])['sha256']==row['sha256']
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',COMMIT+':'+row['path']])
        assert hashlib.sha256(blob).hexdigest()==row['sha256']
check_commit()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y50')
    for evidence in candidate['evidence']:
        assert ref(evidence['path'])['sha256']==evidence['sha256'],'Evidence changed: '+evidence['path']
    queue=monitor.load_queue(str(ROOT))
    if args.operation=='admit':
        assert candidate['stage']=='validation' and candidate['code_commit']==COMMIT
        assert not any(t['id']=='AdaRES-Y50' for group in queue['gpu_queues'].values() for t in group)
        assert monitor.load_experiment(str(ROOT),'AdaRES-Y50') is None
        d=read(prior('y50_diagnosis_1cb1'))
        c=read(prior('y50_production_cpu_v2_1cb1'))
        n=read(prior('y50_production_native_1cb1'))
        b=read(prior('y50_production_controls_v2_1cb1'))
        s=read(prior('y50_production_stress_1cb1'))
        binding='pipeline/analyses/y50_commit_binding_7548_'+TOKEN+'.json'
        prelaunch='pipeline/analyses/y50_prelaunch_7548_'+TOKEN+'.json'
        assert read(binding)['code_commit']==COMMIT and read(prelaunch)['exact_remote_seven_files']
        assert d['maximum_actual_parent_indicator_error']==0 and d['native_nonconstant_expressions']==128
        assert c['parent_stripped_AST_exact'] and c['parent_init_state_exact'] and c['RNG_preserved']
        assert n['actual_tools_train_steps']==2 and n['model_optimizer_scheduler_restore_exact']
        assert all(math.isfinite(v) and v>0 for v in n['new_parameter_updates'].values())
        assert len(b['native_rows'])==8 and all(len(r['gradients'])==4 and all(math.isfinite(v) and v>0 for v in r['gradients'].values()) for r in b['native_rows'])
        cached=[v for r in b['native_rows'] for v in r['cached_controls']]
        assert len(cached)==32 and all(v['repeat_exact'] and v['cyclic_wrong_gap']>0 for v in cached)
        assert max(v['peak_reserved_gib'] for r in s['rows'] for arm in ('parent','candidate') for v in r[arm])<24
        plan=read(prior('y50_post_training_freeze_1cb1'))
        assert plan['protocol']['epochs']==30 and plan['protocol']['seed']==1999
        authorization='pipeline/authorizations/AdaRES-Y50_commit.json'
        assert read(authorization)['kind']=='explicit_human_scoped_commit_authorization'
        review='pipeline/analyses/Y50_authorized_admission_review_7548.md'
        assert not (ROOT/review).exists()
        (ROOT/review).write_text(f'''# Y50 Authorized Admission Review

时间：{datetime.now().astimezone().isoformat()}。用户直接回复“n那你授权啊”，确认前文限定的Y50已测代码提交及既有门控续做；不是自授或扩大授权。
已完成七文件精确提交{COMMIT}，没有包含其他暂存/工作区修改。旧阻塞记录保留，当前提交授权阻塞已解除。

| Gate | Result | 证据范围 |
|---|---|---|
| diagnosis | PASS | 原生指示器边界的等面积/等全Gram32种子控制及固定TRAIN128描述子，不外推整个上游模型盲区或收益 |
| controls | PASS | 独立生产CPU32/final-only/父一致性与原生TRAIN8/32精确缓存/非零错配 |
| native_backward | PASS | 四新增张量原生梯度和真实released-init两步更新；模型/AdamW/PolyLR精确恢复 |
| resources | PASS，测得可运行性 | 66561参数；成对推理/原生压力，预留峰值14.352GiB；不是准确率-资源Pareto优势 |
| independence | PASS | 独立六文件/唯一配置；父字节/AST/初始化/RNG通过，本次local+Git+remote七文件hash一致 |
| constraints | PASS | releasedSPFormer/seed1999/fixed30/128/singleargmax/原loss；server2单GPU1，外来GPU0保留，无validation选择 |

原预注册/所有负面证据/压力模板元数据澄清保留，不重写旧review。文献新颖性、训练效果未知。
冻结9508表达/141场景各两次FIRST主评估、scene-bootstrap10000/1999、TRAINsizecuts1487/3325及子组伤害规则保持不变。
只准入此一个Y50，不制造第二候选或改调度；须通过canonical dispatcher后验证递增native训练步数，PID不等于启动成功。
''',encoding='utf-8')
        decision=monitor.log_decision(str(ROOT),'next_batch','Y50已获得限定提交授权且已测七文件形成真实提交，是否准入同一候选？',
            '仅准入AdaRES-Y50高阶共同覆盖置信度，server2单GPU1，released SPFormer/seed1999/fixed30，保留外来GPU0，不部署诊断权重。',
            '六项既有验收hash复核通过；local/Git/remote七文件相同；人类提交授权明确，解除旧提交阻塞但不放宽科学门槛。',
            str(ROOT/review)+'; '+str(ROOT/binding))
        gates={
            'diagnosis':[ref(prior('y50_diagnosis_1cb1')),ref('pipeline/analyses/y50_diagnosis_prereg_1cb1_'+OLD+'.json'),ref(review)],
            'controls':[ref(prior('y50_production_cpu_v2_1cb1')),ref(prior('y50_production_controls_v2_1cb1')),ref(review)],
            'native_backward':[ref(prior('y50_production_native_1cb1')),ref(prior('y50_production_controls_v2_1cb1')),ref(review)],
            'resources':[ref(prior('y50_production_stress_1cb1')),ref(prior('y50_production_controls_v2_1cb1')),ref(prelaunch),ref(review)],
            'independence':[ref(prior('y50_production_cpu_v2_1cb1')),ref(prior('y50_production_bundle_1cb1')),ref(binding),ref(prelaunch),ref(review)],
            'constraints':[ref('pipeline/goal.json'),ref(authorization),ref(prior('y50_post_training_freeze_1cb1')),ref(review)]}
        receipt='pipeline/admissions/AdaRES-Y50.json'
        assert not (ROOT/receipt).exists()
        (ROOT/receipt).parent.mkdir(exist_ok=True)
        tick.save(ROOT/receipt,dict(id='AdaRES-Y50',candidate_id='AdaRES-Y50',config=CONFIG,code_commit=COMMIT,
            goal_sha256=ref('pipeline/goal.json')['sha256'],decision=ref(Path(decision).relative_to(ROOT).as_posix()),
            gates={name:dict(passed=True,evidence=evidence) for name,evidence in gates.items()},
            authorized_at=datetime.now().astimezone().isoformat(),scientific_limitations='Admission permits a falsifiable fixed-protocol training run, not proven efficacy or novelty.'))
        task=dict(id='AdaRES-Y50',label='Y50 joint-incidence confidence fixed30',config=CONFIG,status='waiting',gpu='1',
            depends_on=[],code_commit=COMMIT,admission_receipt=receipt,patience=30,min_epochs=30,fixed_epochs=30,
            requires_exclusive_ownership=True,scientific_candidate_id='AdaRES-Y50')
        closure.admission(ROOT,task)
        candidate.update(stage='admission',admission_receipt=receipt,next_action='Canonical server2 GPU1 dispatch only after fresh free-card checks, exact admitted commit sync, and native increasing-step verification; never restart blindly.')
        candidate['evidence'].extend([ref(review),ref(receipt),ref(Path(decision).relative_to(ROOT).as_posix())])
        tick.save(ROOT/'pipeline/research_work.json',work)
        queue['gpu_queues']['server-2'].append(task)
        monitor.save_queue(str(ROOT),queue)
        print(json.dumps(dict(admitted=True,id=task['id'],code_commit=COMMIT,receipt=receipt,decision=decision,status=task['status']),indent=2))
    else:
        tasks=queue['gpu_queues']['server-2']
        assert len(tasks)==1 and tasks[0]['id']=='AdaRES-Y50' and tasks[0]['status']=='waiting'
        task=tasks[0]
        assert candidate['stage']=='admission'
        closure.admission(ROOT,task)
        server=monitor.load_servers(str(ROOT))['server-2']
        assert server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
        assert monitor.ssh(server['host'],server['user'],'hostname').strip()=='dl01-MS-7C60'
        pids=monitor.ssh(server['host'],server['user'],"nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits").strip()
        assert not pids,'GPU1 has an existing compute process; leave admitted task waiting'
        events=[]
        started=datetime.now().astimezone().isoformat()
        monitor._process_queue(queue,'server-2',server['host'],server['user'],str(ROOT),set(),{'0','1'},events,
            max_tasks_per_gpu=1,allow_external_overlay=False,remote_code_root=server['remote_code_root'],current_tasks=0,max_tasks_total=2)
        queue=monitor.load_queue(str(ROOT))
        task=next(t for t in queue['gpu_queues']['server-2'] if t['id']=='AdaRES-Y50')
        record='pipeline/analyses/y50_dispatch_7548_'+TOKEN+'.json'
        assert not (ROOT/record).exists()
        tick.save(ROOT/record,dict(complete=True,id='AdaRES-Y50',observed_start_local=started,observed_end_local=datetime.now().astimezone().isoformat(),
            canonical_dispatch=True,task=task,events=events,code_commit=COMMIT,
            limitation='A launched PID alone is not success; require native increasing steps and launch_verified.'))
        candidate['evidence'].append(ref(record))
        if task.get('status')=='running' and task.get('launch_verified') is True:
            candidate.update(stage='training',experiment_id='AdaRES-Y50',next_action='Continue SAME released-init single-GPU fixed30 run without tuning or diagnostic weights; monitor native steps/checkpoints and use frozen paired evaluation after terminal state.')
        else:
            candidate.update(stage='deployment',next_action='Reconcile exact config/PID/log and native progress from existing dispatch; do not blindly restart. Admission/dispatch artifact preserves actual result.')
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(path=record,stage=candidate['stage'],task=task,events=events),ensure_ascii=False,indent=2))
finally:lock.close()
