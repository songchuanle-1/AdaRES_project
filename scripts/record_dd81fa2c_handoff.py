"""Persist this tick's verified handoff and bounded memory before official finish."""
import copy
from datetime import datetime,timezone
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
import closure
import gpu_occupation

TOKEN='dd81fa2cdb754585ab9d4ab0da627f70'

def main():
    lease(TOKEN,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');get=lambda cid:next(c for c in work['candidates'] if c['id']==cid)
        context=tick.load(ROOT/'pipeline/headless/.tick_context.json')
        queue=monitor.load_queue(str(ROOT));running=monitor.load_running(str(ROOT))
        tasks={t['id']:t for ts in queue['gpu_queues'].values() for t in ts}
        before_q,before_r=ref('pipeline/queue.json'),ref('pipeline/running.json')
        observations={}
        for cid,name,gpu,pid in [('AdaRES-Y72','server-1','0',311445),('AdaRES-Y73','server-2','1',62829),('AdaRES-Y74','server-2','0',49891)]:
            task=tasks[cid];assert task['status']=='running' and task['launch_verified'] and task['pid']==pid
            assert running[name][gpu]==cid
            closure.admission(ROOT,task)
            c=get(cid);r=c['latest_training_observation'];v=tick.load(closure.artifact(ROOT,r))
            assert v['token']==TOKEN and v['pid']==pid and v['native_progress_confirmed']
            assert r['path'].endswith('_final.json')
            observations[cid]=r
        occupations={}
        for name in ('server-2','server-1'):
            r=ref('pipeline/checks/gpu_occupation_'+name+'_'+TOKEN+'.json');v=tick.load(closure.artifact(ROOT,r))
            assert v['token']==TOKEN and (datetime.now(timezone.utc)-datetime.fromisoformat(v['observed_at'])).total_seconds()<300
            c=next(c for c in work['candidates'] if c.get('stage')=='blocked' and c.get('blocker_kind')=='exclusive_gpu_slot' and c.get('required_server')==name)
            snapshot_server=next(s for s in context['queue']['servers'] if s['name']==name)
            state=gpu_occupation.validated_state(ROOT,context,c,snapshot_server,queue)
            assert not state['idle_gpu_ids']
            occupations[name]=r
        paused=tasks['AdaRES-Y57'];assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        assert ref('pipeline/goal.json')['sha256']=='4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        y71=get('AdaRES-Y71');terminal=tick.load(closure.artifact(ROOT,y71['terminal_evidence']))
        assert monitor.load_experiment(str(ROOT),'AdaRES-Y71')['status']=='done_complete'
        assert terminal['completed_epochs']=={k:list(range(1,31)) for k in ('Train','Val')}
        y74=get('AdaRES-Y74');smoke=tick.load(closure.artifact(ROOT,y74['native_smoke']))
        assert smoke['steps']==smoke['optimizer_steps']==2 and smoke['shared_head_calls']==14
        freeze=tick.load(closure.artifact(ROOT,y74['post_training_freeze']));parent=tick.load(closure.artifact(ROOT,freeze['parent_protocol']))
        restored=copy.deepcopy(freeze['evaluation']);restored['methods'][0]='AdaRES-Y64'
        assert restored==parent['evaluation'] and freeze['training']==parent['training']
        for cid in ('AdaRES-Y75','AdaRES-Y76'):
            c=get(cid);assert c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' and c.get('implementation_test_passed')
            for field in ('source_manifest','CPU_native_observer_verification','native_smoke_plan_server2_gpu0','native_smoke_plan_server2_gpu1'):
                closure.artifact(ROOT,c[field])
        y76=get('AdaRES-Y76')
        out=ROOT/('pipeline/checks/resource_handoff_'+TOKEN+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=TOKEN,observed_at=datetime.now().astimezone().isoformat(),
            original_live_training=observations,occupation=occupations,
            Y71_terminal=y71['terminal_evidence'],Y71_original_freeze=y71['post_training_freeze'],Y74_smoke=y74['native_smoke'],Y74_receipt=ref('pipeline/admissions/AdaRES-Y74.json'),
            Y74_original_full_protocol_preserved=True,Y76_CPU_version=y76['source_manifest'],Y76_CPU_native_callback_verification=y76['CPU_native_observer_verification'],
            source_HEAD='3c91105459f773326910b1eeaf16450e2d6b90ac',new_candidates=1,new_admissions=1,new_dispatches=1,archived=['AdaRES-Y71'],
            prepared_native_pending=['AdaRES-Y75','AdaRES-Y76'],Y57_pause_preserved=True,formal_FIRST_metrics_updated=False,
            conclusion='Y71 original fixed30 fully verified/archived; actual Y74 native two-step PASS/admission/gated launch/progress; Y76 CPU implementation/exact callback preparation complete. All authorized cards remain busy.'))
        for cid in ('AdaRES-Y71','AdaRES-Y74','AdaRES-Y75','AdaRES-Y76'):
            get(cid)['evidence'].append(ref(out))
        tick.save(ROOT/'pipeline/research_work.json',work)
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+TOKEN+'.md')
        assert not backup.exists();backup.write_bytes(memory.read_bytes())
        phases={cid:tick.load(ROOT/r['path'])['after']['latest'] for cid,r in observations.items()}
        def evidence(r):return r['path']+' SHA'+r['sha256']
        lines=[
            '# AdaRES_project Research Memory',
            '## Goal And Phase',
            '- 当前token '+TOKEN+'，snapshot2026-10-31T22:49:57.792080+00:00，deadline2026-10-31T23:49:57.792080+00:00不延长；postflight只认官方同token收据。',
            '- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled。experiment_first enabled；snapshot admission2/creation2，本轮实际新增候选1/准入1/启动1/终态归档1。',
            '- 授权server1 GPU0既有350W；server2 GPU0保持280W/GPU1既有390W；每卡一独立实验。Y57实验暂停保留；禁server6/跨项目/改调度/改硬件功耗/驱逐/重启迁移受保护运行。',
            '- goal科学字节SHA4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a不变；最新AGENTS/SERVERS及authorizations/server2_gpu1_20261029.json SHA6b24941bb4e278650fb8964af29657bf1d9f71a0bdd6b58c0f241564cec2d838覆盖旧GPU1资源句；原收据不重签。',
            '- PROJECT_ROOT=D:/project，PYTHONDONTWRITEBYTECODE=1，PYTHONUTF8=1，python -B；密钥SSH可用，未读密码文件，不推定sudo。',
            '- 上轮完整64行记忆精确备份：'+evidence(ref(backup))+'；历史机制/禁用/专用恢复及原证据沿备份保留。',
            '## Champion And Baseline',
            '- 目标mIoU .5150；历史CSV Y36 .5110。冻结FIRST：Y36 .510725438294、W2 .510107995189、B1 .502221764356、Y52 .509708428366、Y54 .506342952986。本轮无正式全量FIRST/成本更新；训练best不替代FIRST或晋升。',
            '- 后评原协议：9508表达/141场景、候选+Y36/W2/B1各两次、FIRST主结果、scene bootstrap10000/seed1999、TRAIN cuts1487/3325、原subgroup harm/mIoU/Acc25/50/activequeries/latency/memory和同卡cost。Y54大目标对Y36伤害CI[-.02678985,-.00070265]与Y52总体跨0保留；单seed不估seed方差。',
            '## Active Hypotheses',
            '- Y74语言增量门控：六原qla输出后/原qqa前，q_out=q_language+tanh(q_visual@W)*(q_language-q_visual)，零W256x256/no RNG，+65536/总164695173。原stage0/static mask/padding/全部输入/现有2D/loss/共享头/SCG/seed1999/fixed30/native YAML保持；不重开Y26 discarded-norm/Y53 padding-leakage，不宣称创新或效益。',
            '- Y74本轮真实server2 GPU0/280W原生YAML/tools.train/TRAIN两步PASS；loss4.615511417388916/6.321258068084717，四原场景16表达，两批134280/350887点；每步540有限梯度/10原头梯度/7头调用/6门控和真实qqa消费、独立公式/首零次响应/有限新梯度与更新；峰6744440832B。不是全量显存或效益保证。',
            '- Y74实际native raw：'+evidence(ref('pipeline/analyses/y74_gpu0_smoke_'+TOKEN+'.json'))+'；14准入/资源/源版本负守卫通过；原Y64完整training/evaluation深复制核等。receipt：'+evidence(ref('pipeline/admissions/AdaRES-Y74.json'))+'；freeze：'+evidence(y74['post_training_freeze'])+'。',
            '- Y75=adares_y75_language_query_init：原query生成后/首次预测前一次q+(q*mean_valid(原投影language))@W，零W65536/总164695173；原sampling/位置/padding/七阶段/所有loss/input/YAML保持。原CPU真实模型/发布backbone/TRAIN162738点8表达、float32/64公式/zero/RNG/梯度/七阶段及真实初始预测消费/bypass通过；不是GPU native PASS。',
            '- Y75保留commit9bf15a255ed2c960429ecf57ab47de05c484b631，branch refs/heads/adares-y75-language-query-init，parent=3c；独立W2机制，不组合Y74。原CPU callback verification e042轮保留；本轮四份Y75/Y76计划及源码/goal/commit逐项复核有效，不重跑原准备。',
            '- Y75 GPU0计划：'+evidence(get('AdaRES-Y75')['native_smoke_plan_server2_gpu0'])+'；GPU1计划：'+evidence(get('AdaRES-Y75')['native_smoke_plan_server2_gpu1'])+'。原body scripts/probe_y75_native_smoke.py SHAd3b5f47e22f87ff26df050b76cb3e40e5c7efdff6d1ab95c4ed63c571019a6ed。',
            '- Y76=adares_y76_stage_confidence_conditioning：仅原共享QueryCompetitionIndicator query_norm前q+offset[stage0..6]，零offset[7,256]/no RNG，+1792/总164631429。保留一共享头/原mask-score/重叠/targets/loss/selector/所有输入/seed1999/fixed30/YAML。与Y60独立头、Y67 mask-score affine不同；不重开原实验，不宣称新增信息/创新/效益。',
            '- Y76实际CPU原生完整模型/发布backbone/原TRAIN162738点8表达通过；float32/64逐标量stage公式、非法stage拒绝、稀疏stage梯度、joint permutation/零输出梯度/RNG、完整DEC七阶段W2输出/输入梯度锚点、原七头query-stage对象/真实query_norm消费、七行有限新梯度、非零只改对应confidence而全部mask/score精确不变/bypass通过。raw：'+evidence(ref('pipeline/analyses/y76_cpu_'+TOKEN+'.json'))+'。',
            '- Y76仅七测试A文件以隔离index保留commitc3849ead0b8e4f9320d3e353d561016b3b495ce1，branch refs/heads/adares-y76-stage-confidence-conditioning，parent=Y75/9bf；主HEAD3c/index/queue/running保持。version：'+evidence(y76['source_manifest'])+'。',
            '- Y76实际冻结native回调在两CPU合成DEC原optimizer步骤通过：原七头query+stage/未detach/真实query_norm消费/static mask/padding身份、首零次响应/独立公式、每步387有限梯度张量/10原头梯度/全部七offset行有限非零梯度与更新。verification：'+evidence(y76['CPU_native_observer_verification'])+'；明确无CUDA/native sparse task/GPU PASS/准入。',
            '- Y76两原生GPU计划及11证据拒绝守卫、Python3.8 AST、goal/七源码YAML/commit一致、stale-goal/tampered-source拒绝通过。GPU0：'+evidence(y76['native_smoke_plan_server2_gpu0'])+'；GPU1：'+evidence(y76['native_smoke_plan_server2_gpu1'])+'。body：'+evidence(ref('scripts/probe_y76_native_smoke.py'))+'。',
            '- Y75先于Y76；二者均blocked/validation/exclusive_gpu_slot/server2 GPU0，GPU1备选。只在canonical原slot owner终态归档+fresh物理idle/no compute/no reservation后选精确计划真实native两步；PASS后当轮原完整freeze/准入/精确additive版本选择与新存活source契约/门控启动。不要重放已完成CPU/version/plans/callback；server1须另备精确private runtime/path/350W备用计划。',
            '- Y71原3D与现有2D融合前(s*a)@W再原sum_norm，零W65536；原PID53714本轮自然退出。完整Train/Val1..30、稳定finite last30/best/model/optimizer544/scheduler30/19及原源/YAML/receipt/freeze已核；20生命周期守卫通过，canonical done_complete/30/30归档。训练best .4988786280155182在epoch19，不是FIRST。',
            '- Y71终态proof：'+evidence(y71['terminal_evidence'])+'；last SHA d5269e41a7c7dec7d9b2885d2c1cb43e0ff1dce7057dd4ef235d40e585e5dd88，best SHA2b55d76acc7ddc8a0cd2c570d214d1fbede986055d23445905275246b6f7a224，远端原目录.../AdaRES_checkpoints/adares_y71_visual_product_fusion_res/20261030_191128保留。原完整日志在Checkpoints/server-2/<basename>/'+TOKEN+'/fixed30.log；finalizer不得重放。',
            '- Y72六阶段current visual投影细化静态mask特征、Y73六阶段初始query锚点各65536；原机制/CPU/native/完整后评freeze保持。Y72 code c60、Y73 code7da，原receipt/PID不改。',
            '## Closed Or Failed Directions',
            '- 保留Y53/Y55/Y50/Y51/Y49rev1/Y46/Y48等拒绝及goal bans；F1否决mask-entropy，禁val扫描entropy/margin/occupancy/threshold/temperature/loss/feature subsets/epoch数，不改名重开。旧Y22/Y23、所有专用恢复及负结果见AGENTS和备份。',
            '- Y76 CPU callback初封装错用native.build_optimizer、attempt2漏定义原dataloader；两个失败.py/.log完整保留并绑定verification。独立attempt3仅修gorilla优化器入口并提供真实dataloader依赖，原模型/冻结native body/plans无变化；CPU回调真实通过。失败scope=implementation wrapper，不是否定机制或GPU PASS。',
            '- server1私有torch栈/CUDA11.3库及精确数据/路径证明沿原证据；不改系统或其他环境。Y56-Y71已归档产物不重训、不重放终态helper，Y57继续暂停。',
            '## Running And Queue',
            '- 最新真实native阶段：'+json.dumps(phases,ensure_ascii=False)+'；三PID311445/62829/49891真实递增、独占/功耗/source/YAML/receipt均核；waiting0。',
            '- 当前观察：'+ '; '.join(cid+' '+evidence(r) for cid,r in observations.items())+'。',
            '- 主master7da→3c仅选择七已保留且实际GPU测试的Y74 A文件，其余index条目与工作文件保持；server2生产3c仅additive追加Y74，原Y73源码/YAML/receipt/PID保持。server1生产仍c60。',
            '- 现用scripts/y74_deployment.py observe --id AdaRES-Y72|AdaRES-Y73|AdaRES-Y74 --token NEW_TOKEN --label progress；仅接受明确87019/c60/7da/3c边与精确各七A文件、原源码/YAML/PID/功耗和native增长。旧y72_y73_deployment/更旧observer当前HEAD不兼容，留历史勿再调用。后续Y75/Y76部署前须新建精确契约，不能放行任意marker或改旧hash-bound文件。',
            '- 15项fixed30已完成Y56/Y58/Y59/Y60/Y61/Y62/Y63/Y64/Y65/Y66/Y67/Y68/Y69/Y70/Y71等待各自原完整FIRST后评/cost，保留required server/GPU。Y71新增blocked/validation/frozen_evaluation/server2GPU0，现Y74训练占用。',
            '- canonical占用两台确认所有授权卡busy+reserved；finish须<300秒，超时新label保留旧证据。'+ '; '.join(name+' '+evidence(r) for name,r in occupations.items())+'。',
            '- server2=192.168.1.163/dl01-MS-7C60，root=/bak_home/songcl/Project/AdaRES_project，code=.../AdaRES，python=/home/songcl/.conda/envs/adares_project/bin/python，原data=/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/。',
            '- server1=192.168.1.175/dl02-MS-7B73，root=/data1/songcl/Project/AdaRES_project，private python=.../runtime/adares_project/bin/python，sitepackages=.../lib/python3.8/site-packages，pointnet=.../runtime/pointnet2-0.0.0-py3.8-linux-x86_64.egg，LD_LIBRARY_PATH=.../runtime/cuda113lib64，data=.../data/，work=.../AdaRES_checkpoints。exact-data proof checks/server1_data_verified_63b27dd8efe04a0bab6dc0291cefdbf0.json SHA8c590fa828fbc6c0df840c21ba88b1802cc0d2323eb0a124a57aa328b0c2361f；只允许SERVERS两path mappings。',
            '- Y57 paused/dispatch_hold/resume_requires_user_authorization，原20261021_004739/last.pth epoch27/model/optimizer543/scheduler27/best .5028846859931946保留；仅明确Y57继续授权才能从28续同模型/优化器/调度器至fixed30，不迁移/不从pretrain重启。',
            '## Open Questions And Risks',
            '- 尚无本轮正式全量复评/成本或科学胜出结论；seed方差与server2掉卡/散热/Xid79根因仍未知。真实训练增长不是硬件修复证明；原恢复要求保持。',
            '- 二步native PASS只保证已测批次可执行；不代替效益、完整数据显存或original full freeze。Y75/Y76还无GPU任务PASS，不能直接入队。所有资源hold须当前token/snapshot/原始query哈希与<300秒真实占用。',
            '- heartbeat保持原owner和原2小时频率，不改自动化/通知，不建线程/嵌套codex exec；没有新操作授权缺口，不重复索权。',
            '## Last Tick',
            '- '+TOKEN+'：先核原训练，完成并归档Y71；fresh独占GPU0原native检查→保存测试版本→原完整后评freeze与准入→canonical启动Y74→真实native增长；保留Y72/Y73/Y57。',
            '- 待跑不足新增Y76，实际CPU原生模型/发布backbone/真实TRAIN加载及独立控制、隔离保留版本、原生双卡计划与11守卫、真实CPU callback两步完成；保留封装失败原证据，不声称native GPU或效益。两储备Y75/Y76仍等待GPU。',
            '- 当前handoff：'+evidence(ref(out))+'；original frozen evaluation等待三卡原训练自然释放，优先下一空槽的已准备native候选；不提前停止、不重训、不放宽协议。',
            '- 接着只用官方report/finish并核FINALIZED/PASS/context_committed/released和锁/pending/archive_pending。此记忆不预签结果；释放后不写。'
        ]
        assert len(lines)<120 and sum(line.startswith('## ') for line in lines)==7
        memory.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        notes=ROOT/('pipeline/checks/notes_'+TOKEN+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=[
            'Y71自然完成固定30轮；原完整Train/Val1..30与稳定finite last30/best/model/optimizer/scheduler、source/YAML/原receipt/freeze核验，canonical done_complete归档。训练best0.498878628仅为训练监控；正式FIRST未更新。',
            '释放的server2 GPU0保持280W并fresh核独占空闲，Y74原生YAML/tools.train真实TRAIN两步loss4.615511/6.321258、全部六门控与真实qqa消费、有限梯度/实际更新通过；14负守卫、原Y64完整后评协议核等、准入与安全additive部署，已门控启动且native计数递增。',
            'Y72/Y73原PID/源/配置/收据和训练保留；新精确observer覆盖已注册87019/c60/7da/3c及七文件版本边。Y57实验暂停继续保留。',
            '新增Y76置信度阶段输入1792参数；真实CPU原生模型/发布backbone/TRAIN162738点8表达、独立公式/零/RNG/七阶段W2锚点/梯度/真实消费/仅对应confidence改变通过；七文件独立保留，双卡native计划/11证据负守卫和实际CPU callback两步通过。两封装依赖失败原证据保留，原模型/native计划未改；未声称GPU任务PASS。',
            '两台当前canonical occupation确认三授权卡busy且预约；Y75/Y76原native检查与15项完整冻结后评/cost等待独占资源，已保存当前状态和七段记忆。'],decisions=[
            '本轮实际新增候选1/准入1/启动1/完成1，均在snapshot budget2内；有真实native PASS即本轮入队启动。当前资源被三项既有训练占用，停止新增工作并保留两项储备。',
            '保留原数据/验证边界/全部输入/seed1999/fixed30/禁用项和科学goal字节；不把训练best或CPU回调当正式FIRST、GPU PASS或效益。',
            'server1 GPU0与server2 GPU0/1一卡一实验和既有350/280/390W不变；Y57续训仍须单独明确请求；不触碰server6、其他项目、调度频率。']))
        assert ref('pipeline/queue.json')==before_q and ref('pipeline/running.json')==before_r
        print(json.dumps(dict(handoff=ref(out),memory=ref(memory),memory_lines=len(lines),backup=ref(backup),notes=str(notes),phases=phases)))
    finally:lock.close()

if __name__=='__main__':main()
