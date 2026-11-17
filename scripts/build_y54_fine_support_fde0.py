"""Mechanically isolate W2; reconnect only gathered fine-support validity."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
from register_y53_padding_diagnosis_fde0 import ref, TOKEN
import foreground_tick as tick
import pipeline_monitor as monitor


class RemoveOnlySupportRepair(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == 'fine_batch_mask' for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'get_scg_weight' and any(k.arg == 'pos_k' and isinstance(k.value, ast.Name) and k.value.id == 'inst_feats_fine_pos' for k in node.keywords):
            node.keywords = [k for k in node.keywords if k.arg != 'k_mask']
        if isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'qia_layers_fine':
            node.keywords = [k for k in node.keywords if k.arg != 'batch_mask']
        return self.generic_visit(node)


def main():
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds() > 700
        for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
            assert ref('pipeline/headless/' + name)['sha256'] == owner[key]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y54')
        assert candidate['stage'] == 'diagnosis'
        for row in candidate['evidence']:
            assert ref(row['path']) == row
        diagnostic_path = 'pipeline/analyses/y54_fine_support_native_fde0_' + TOKEN + '.json'
        diagnostic = tick.load(ROOT / diagnostic_path)
        assert diagnostic['complete'] and diagnostic['actual_sparse_forward_and_FPS']
        assert diagnostic['native_padding_attention_mass'] > 0
        assert diagnostic['native_block_padding_effect_max'] > diagnostic['original_block_repeat_max'] == 0
        assert diagnostic['corrected_block_padding_effect_max'] == 0
        assert diagnostic['full_decoder_indicator_effect_max'] > 0 and diagnostic['all_valid_negative_control_indicator_max'] == 0
        source_dir = ROOT / 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
        destination = ROOT / 'AdaRES/ipdn/model_paper5/adares_y54_fine_support'
        config_source = ROOT / 'AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
        config_target = ROOT / 'AdaRES/configs_song/model_paper5/adares_y54_fine_support_res.yaml'
        assert not destination.exists() and not config_target.exists()
        sources = sorted(source_dir.glob('*.py'))
        assert len(sources) == 6
        old_tree = ast.parse((source_dir / 'dec.py').read_text(encoding='utf-8'))
        new_tree = copy.deepcopy(old_tree)
        decoder = next(n for n in new_tree.body if isinstance(n, ast.ClassDef) and n.name == 'DEC')
        forward = next(n for n in decoder.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_iter_pred')
        loop = next(n for n in forward.body if isinstance(n, ast.For))
        position = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inst_feats_fine_pos' for t in n.targets))
        loop.body[position+1:position+1] = ast.parse('fine_batch_mask = batch_mask.gather(dim=1, index=sample_inds_fine)').body
        geometry = [n for n in ast.walk(forward) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'get_scg_weight' and any(k.arg == 'pos_k' and isinstance(k.value, ast.Name) and k.value.id == 'inst_feats_fine_pos' for k in n.keywords)]
        fine = [n for n in ast.walk(forward) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == 'qia_layers_fine']
        assert len(geometry) == len(fine) == 1
        geometry[0].keywords.append(ast.keyword(arg='k_mask', value=ast.Name(id='fine_batch_mask', ctx=ast.Load())))
        fine[0].keywords.append(ast.keyword(arg='batch_mask', value=ast.Name(id='fine_batch_mask', ctx=ast.Load())))
        repaired = ast.unparse(ast.fix_missing_locations(new_tree)) + '\n'
        compile(repaired, str(destination / 'dec.py'), 'exec')
        inverse = RemoveOnlySupportRepair().visit(ast.parse(repaired))
        assert ast.dump(inverse, include_attributes=False) == ast.dump(old_tree, include_attributes=False)
        cfg = yaml.safe_load(config_source.read_text(encoding='utf-8'))
        assert cfg['train']['epochs'] == 30 and cfg['train']['seed'] == 1999
        assert cfg['model']['dec']['num_proposal_fine'] == 512 and cfg['model']['sampling_module']['num_proposal'] == 128
        goal = tick.load(ROOT / 'pipeline/goal.json')
        now = datetime.now().astimezone().isoformat()
        review_path = 'pipeline/reviews/motivation_review_Y54_fde0.md'
        design_path = 'pipeline/designs/exp_design_Y54_fde0.md'
        assert not (ROOT / review_path).exists() and not (ROOT / design_path).exists()
        review = '\n'.join([
            '# Motivation Review - AdaRES Y54', '时间：'+now+'；原生诊断支持，尚无训练收益。',
            '## 1. Goal', '当前目标.515，历史CSV冠军Y36 .5110、差.0040。完整目标如下：',
            '```json', json.dumps(goal, ensure_ascii=False, indent=2), '```',
            '## 2. Story Status', '| 假设 | 状态 | 缺口 |', '|---|---|---|',
            '| 语言padding进入前向置信度Y53 | 源码+8TRAIN/32表达反证，拒绝 | 不实现、不重开Y23 |',
            '| 辅助点测度Y52 | 有限原生证据，待限定提交许可 | 未训练、收益未知 |',
            '| 细粒度3D支持有效性Y54 | 原生双场景有真实虚构支持及匹配干预分离 | 独立实现/反向/恢复/资源/训练效果 |',
            '## 3. Experiment Assessment', 'Y50当前固定30实现因相对W2中目标伤害规则拒绝；不由总体均值的微小基线差异宣称收益。Y53的新反证不是历史Y23损失支持问题的重新评估。',
            '## 4. Gap Analysis', '可信置信度需要证据来自真实3D支持，不能依赖另一场景造成的padding。当前全有效表达负控制变化0，但缺少独立原型的原生反向和可执行门禁。',
            '## 5. Design Rationale', '只把现有batch_mask沿native fine gather传递到空间邻域和fine attention，排除不存在的支持；不增加头/权重、不修改语言/匹配/损失或512/128超参数，不与Y52叠加。',
            '## 6. Next Batch', '仅同一Y54独立七文件准备，见对应实验设计；通过全部门禁且获限定提交许可之前不入队。原生诊断：'+diagnostic_path,
        ]) + '\n'
        (ROOT / review_path).write_text(review, encoding='utf-8', newline='\n')
        design = '\n'.join([
            '# Experiment Design - AdaRES Y54', '时间：'+now+'；状态：实现准备，不是准入。',
            '## 1. Motivation', goal['description'], '角色：检验去除批内虚构3D支持是否能使共享query学习及置信度证据更一致；不承诺提升。',
            '## 2. Current Best Results', '| ID | 状态 | mIoU | Acc50 | spmIoU |', '|---|---|---|---|---|',
            '| Y36 | 历史CSV完成 | .5110 | 未知 | 未知 |', '| W2 | 上轮全量冻结FIRST | .5101051604 | .5629995793 | .4737185291 |',
            '| B1 | 上轮全量冻结FIRST | .5021216787 | .5527976441 | .4666861819 |', '| Y54 | 未训练 | 未知 | 未知 | 未知 |',
            '## 3. Analysis & Direction', '实际原四组TRAIN配对有3600个跨层虚构fine支持槽、34172个可见query-key边，非独立样本数。虚构支持注意力质量为正；原样repeat0，padding干预作用12.5120，正确mask后0；全有效表达置信度负控制0。不能外推训练收益或普遍数值界。',
            '## 4. Experiment Plan', '| ID | Config | 假设 | 下一门禁 |', '|---|---|---|---|',
            '| AdaRES-Y54 | configs_song/model_paper5/adares_y54_fine_support_res.yaml | native fine支持与邻域继承有效mask | 独立原型、实际TRAIN反向/恢复和资源 |',
            '## 5. GPU Allocation Plan', '仅server-2 GPU0/1；GPU0外来工作保留。诊断需要实时空闲和独占租约，不预留卡。提交/准入前不派发训练。',
            '## 6. Implementation Checklist', '- [x] 原生实际支持审计及matched干预/正负/重复控制。', '- [ ] 独立六文件包+唯一basename YAML；仅DEC.forward_iter_pred三项mask连接，逆向AST相等审计。', '- [ ] 实际原生反向、非空优化器/调度/RNG完整恢复、资源和同权重推理对照。', '- [ ] 限定人工提交许可、真实commit与六门禁回执、canonical派发及递增训练步数。',
            '## 7. Risk Assessment', '新增参数预期0仍须实测；padding纠正改变训练上下文，但标准单场景推理无padding时应保持行为。少量输入不证明最坏情况或Pareto；保留潜在空邻域/小支持、数值及原生loss约束。原型不改损失/匹配、不部署诊断W2权重。',
            '## 8. Smell Check', '缺陷证据与因果干预来自native不存在支持；不是扫描mask阈值/温度/权重/轮数，也不把修复包装为新架构或已证新颖性。只恢复原生有效性信息流。',
            '## 9. Prospective Evaluation', '若以后正式准入：released SPFormer/seed1999/fixed30/单GPU/128query/singleargmax和原loss不变。冻结ScanRefer val全9508表达/141scene，Y54/W2/B1/当前冠军Y36各两次且FIRST主结果；scene-bootstrap10000/1999，TRAIN cuts1487/3325，任一参照子组delta区间上界<0触发伤害审查。报告mIoU/Acc25/Acc50/active queries/warm latency/peak memory/parameters。Y36 exact best检查点须在准入前读取唯一既有训练归档并绑定哈希，不按新验证结果择优。无验证调参，单seed区间不覆盖训练seed不确定性。',
        ]) + '\n'
        (ROOT / design_path).write_text(design, encoding='utf-8', newline='\n')
        destination.mkdir()
        files = []
        for source in sources:
            target = destination / source.name
            if source.name == 'dec.py':
                target.write_text(repaired, encoding='utf-8', newline='\n')
            else:
                target.write_bytes(source.read_bytes())
            files.append(ref(target.relative_to(ROOT).as_posix()))
        config_target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        assert yaml.safe_load(config_target.read_text(encoding='utf-8')) == cfg
        files.append(ref(config_target.relative_to(ROOT).as_posix()))
        bundle_path = 'pipeline/analyses/y54_prototype_bundle_' + TOKEN + '.json'
        assert not (ROOT / bundle_path).exists()
        bundle = dict(complete=True, id='AdaRES-Y54', token=TOKEN, created_at=now, files=files,
            source_files=[dict(**ref(p.relative_to(ROOT).as_posix()), normalized_sha256=hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()) for p in sources],
            diagnosis=ref(diagnostic_path), motivation_review=ref(review_path), design=ref(design_path),
            method_scope='Only DEC.forward_iter_pred: gather original validity at native fine indices; pass to spatial neighbor k_mask and fine cross-attention batch_mask. Removing these3AST edits exactly recovers the entire original decoder AST.',
            inverse_AST_equals_parent=True, unchanged_other_five_model_files=True, model_config_value_identical=True,
            new_parameters_expected=0, actual_parameters='pending_native_measurement',
            code_status='Independent local uncommitted prototype, not deployed/admitted/trained; diagnostic temp package only under guarded runner.',
            literature_novelty='not established; implementation defect correction, not a novelty claim')
        tick.save(ROOT / bundle_path, bundle)
        for path in (diagnostic_path, 'scripts/build_y54_fine_support_fde0.py', review_path, design_path, bundle_path):
            candidate['evidence'].append(ref(path))
        candidate['stage'] = 'implementation'
        candidate['gate_summary']['diagnosis'] = 'supported_actual_native_invalid_fine_support_attention_and_matched_effect'
        candidate['gate_summary']['controls'] = 'matched_diagnostic_padding_inert_after_correction; all_valid_negative0; independent_model_pending'
        candidate['next_action'] = 'Test SAME frozen independent seven-file prototype on actual native TRAIN: finite fine-path gradients, zero invalid-support source gradient, two native optimizer steps and exact nonempty restore, parameters/resources and paired same-weight predict. Bind exact Y36 comparator archive before any admission. No automatic commit or training.'
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(bundle=ref(bundle_path), files=len(files), inverse_AST_exact=True, stage=candidate['stage']), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
