"""Record bounded native readiness and a genuine scoped human-commit dependency."""
import ast
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
from register_y53_padding_diagnosis_fde0 import ref,TOKEN
import foreground_tick as tick
import pipeline_monitor as monitor


def main():
    lock=monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token']==TOKEN and owner['thread_id']==os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds()>120
        for name,key in (('.tick_context.json','context_sha256'),('.tick_context_state.next.json','pending_sha256')):
            assert ref('pipeline/headless/'+name)['sha256']==owner[key]
        work=tick.load(ROOT/'pipeline/research_work.json')
        candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y54')
        assert candidate['stage']=='validation'
        for row in candidate['evidence']:
            assert ref(row['path'])==row
        bundle_path='pipeline/analyses/y54_prototype_bundle_'+TOKEN+'.json'
        gates_path='pipeline/analyses/y54_native_gates_fde0_'+TOKEN+'.json'
        corner_path='pipeline/analyses/y54_small_support_cpu_fde0_'+TOKEN+'.json'
        diagnostic_path='pipeline/analyses/y54_fine_support_native_fde0_'+TOKEN+'.json'
        bundle,gates,corner,diagnostic=(tick.load(ROOT/path) for path in (bundle_path,gates_path,corner_path,diagnostic_path))
        assert all(value['complete'] for value in (bundle,gates,corner,diagnostic))
        assert gates['actual_independent_package'] and gates['parameter_delta']==0
        assert gates['parameter_counts']['candidate']==gates['parameter_counts']['parent']==164629637
        assert all(gates['nonempty_optimizer_restore'].values()) and len(gates['optimizer_steps'])==2
        assert all(row['finite_backward'] for row in gates['native_backward'])
        actual_padding=[layer for row in gates['native_backward'] for layer in row['layers'] if layer['padded_fine_supports']>0]
        assert actual_padding and all(row['padding_source_gradient_max_abs']==0 and row['valid_source_gradient_max_abs']>0 for row in actual_padding)
        assert corner['case_count']==5 and all(row['backward_finite'] and row['all6neighborhoods_nonempty'] and row['padded_support_gradient_max_abs']==0 for row in corner['rows'])
        assert gates['native_training_peak_reserved_bytes']<24*2**30
        parity=[]
        for row in gates['inference_parity']:
            repeats=row['parent_repeat']+row['candidate_repeat']
            keys=('binary_mask_points_changed','selected_query_changes','maximum_expression_iou_difference')
            within=all(row['matched'][key]<=max(value[key] for value in repeats) for key in keys)
            parity.append(dict(scene=row['scene'],matched=row['matched'],within_measured_behavioral_repeats=within))
        assert all(row['within_measured_behavioral_repeats'] for row in parity), 'Inference requires investigation, not automatic readiness'
        for row in bundle['files']:
            assert ref(row['path'])==row
            if row['path'].endswith('.py'):
                tree=ast.parse((ROOT/row['path']).read_text(encoding='utf-8'))
                modules=[node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
                assert not any('adares_' in value and 'adares_y54_fine_support' not in value for value in modules)
        paths=[row['path'].removeprefix('AdaRES/') for row in bundle['files']]
        tracked=subprocess.check_output(['git','-C',str(ROOT/'AdaRES'),'ls-files','--']+paths,text=True).strip()
        assert not tracked
        assert not (ROOT/'pipeline/authorizations/AdaRES-Y54_commit.json').exists()
        authorization='pipeline/authorizations/AdaRES-Y50_commit.json'
        assert tick.load(ROOT/authorization)['id']=='AdaRES-Y50'
        assert gates['unique_existing_Y36_best'] and len(gates['champion_Y36_archive_intake'])==1
        y36=gates['champion_Y36_archive_intake'][0]
        assert '/adares_y36_local_adjacency_res/' in y36['path'] and y36['path'].endswith('/best.pth')
        freeze_path='pipeline/analyses/y54_post_training_freeze_'+TOKEN+'.json'
        review_path='pipeline/analyses/y54_native_gate_review_'+TOKEN+'.json'
        assert not (ROOT/freeze_path).exists() and not (ROOT/review_path).exists()
        now=datetime.now().astimezone()
        freeze=dict(complete=True,id='AdaRES-Y54',token=TOKEN,preregistered_at=now.isoformat(),tested_bundle=ref(bundle_path),goal=ref('pipeline/goal.json'),design=ref('pipeline/designs/exp_design_Y54_fde0.md'),
            training='Released SPFormer backbone/seed1999/fixed30/single authorizedGPU/128queries/singleargmax, unchanged native W2 loss and optimizer schedule. No diagnostic W2 weights used to initialize formal training. Canonical best from completed fixed30 run, not new epoch/model/threshold selection.',
            protocol=dict(dataset='ScanRefer validation,9508expressions/141scenes',primary='FIRST full evaluation mIoU',evaluation='Y54/W2/B1/Y36 each twice; FIRST primary, preserve independent repeat noise',bootstrap=dict(unit='whole scene clusters',expression_weighted=True,draws=10000,seed=1999),size_cuts_from_TRAIN=[1487,3325],material_harm='For any small/medium/large subgroup versus ANY reference, paired scene-bootstrap95%CI upper<0 triggers harm review; no posthoc threshold or cut selection',report=['mIoU','Acc25','Acc50','active_queries','warm_latency','peak_memory','parameters'],lightweight='Require measured native feasibility with no added parameters; compare trained accuracy/cost versus parent/champion/baseline; do not claim global Pareto from bounded pretraining probes'),
            comparators=dict(Y36=dict(**y36,source_manifest=ref('pipeline/analyses/y36_source_bundle_lf_8fd5e469dee241da8f21da7c870c0d3f.json'),config='configs_song/model_paper5/adares_y36_local_adjacency_res.yaml'),W2=dict(path='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth',sha256='e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52',config='configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'),B1=dict(path='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b1_standard_res/20260918_010854/best.pth',sha256='cfc89ec0dc64f240913968f307d19c623f5ff0dc51c8e61de64e35abea05dd8b',config='configs_song/model_paper5/adares_b1_standard_res.yaml')),
            evaluator=ref('AdaRES/tools/evaluate_adares_frozen_full.py'),limitations='One training seed; clustered CI is not seed uncertainty. Recheck all source/checkpoint/goal/commit bindings before dispatch/evaluation; no receipt or training authorization created by this plan.')
        tick.save(ROOT/freeze_path,freeze)
        profile={role:{row['scene']:row for row in gates['inference_profile'] if row['role']==role} for role in ('parent','candidate')}
        ratios={scene:profile['candidate'][scene]['median_seconds']/profile['parent'][scene]['median_seconds'] for scene in profile['parent']}
        costs=dict(parameters=gates['parameter_counts'],parameter_delta=0,native_training_peak_reserved_bytes=gates['native_training_peak_reserved_bytes'],native_training_peak_reserved_GiB=gates['native_training_peak_reserved_bytes']/2**30,stress_measured_median_seconds=statistics.median(row['seconds'] for row in gates['stress'] if not row['warmup']),paired_scene_latency_ratios=ratios,median_scene_latency_ratio=statistics.median(ratios.values()),inference_allocated_equal=all(profile['candidate'][scene]['peak_allocated_bytes']==profile['parent'][scene]['peak_allocated_bytes'] for scene in profile['parent']),limitation='Timing covers frozen TRAIN scenes,2warmups/3measures each; not full latency distribution, reliable speedup or a global Pareto guarantee.')
        gate_summary=dict(diagnosis='supported_actual_native_fine_padding_attention_and_control_separated_intervention',controls='native_same_input_corrected_padding_effect0/all_valid_negative0; exact_independent_small_support5cases',native_backward='actual4nativebatch2/32expressions/all6fine_layers finite nonzero valid+attention+indicator gradients; invalid_source_grad0',resources='parameters_delta0/native_reserved<24GiB/paired_predict_cost_recorded; bounded_scope_only',independence='independent_six_file_package+unique_basename_config; inverse3edit_AST_exact; actual_state_schema_equal',constraints='goal/bans/fixed30/no_validation_tuning respected; four_method_posttraining_plan_frozen; real_commit_and_receipt_pending')
        review=dict(complete=True,id='AdaRES-Y54',token=TOKEN,reviewed_at=now.isoformat(),bundle=ref(bundle_path),diagnosis=ref(diagnostic_path),native_gates=ref(gates_path),small_support_controls=ref(corner_path),post_training_freeze=ref(freeze_path),gate_summary=gate_summary,inference_parity=parity,cost=costs,model_revision_count=0,scientific_state='Supported actual implementation defect and bounded executable native controls; trained Y54 mIoU/benefit/novelty/Pareto unknown.',code_state='Exact seven tested files remain uncommitted; no automatic Git add/commit.',existing_authorization_Y50_only=ref(authorization),admission_receipt_created=False,training_launches=0,required_human_action='Explicit authorization ONLY for the seven exact Y54 files in tested_bundle, or supply their exact commit. Then revalidate current goal/source/gates/commit and use existing dispatcher with live idle checks and normal Y50 terminal reconciliation. Do not extend Y50 authorization, commit other files, or deploy diagnostic weights.')
        tick.save(ROOT/review_path,review)
        for path in (corner_path,freeze_path,review_path,'scripts/record_y54_gate_review_fde0.py'):
            if path not in {row['path'] for row in candidate['evidence']}:
                candidate['evidence'].append(ref(path))
        candidate['stage']='blocked'
        candidate['blocker_kind']='human_authorization'
        candidate['blocker']='All currently planned independent Y54 native/edge/resource/restore controls completed within documented scope; seven tested files remain uncommitted. Canonical contract forbids automatic Git add/commit, and existing explicit authorization is Y50-only, not Y54. This is NOT GPU shortage or a trained-benefit claim.'
        candidate['retry_after']=(now+timedelta(hours=4)).isoformat()
        candidate['unblock_condition']=review['required_human_action']
        candidate['next_action']=review['required_human_action']
        candidate['gate_summary']=gate_summary
        candidate['wake_paths']=['pipeline/authorizations/AdaRES-Y54_commit.json',bundle_path]
        candidate['native_review']=ref(review_path)
        candidate['post_training_freeze']=ref(freeze_path)
        candidate['attempted_alternatives']=['Source audit and exact coarse/fine support-position matching on native sparse/FPS TRAIN inputs.','Parent/repeat/invalid-source/corrected/valid-positive and all-valid negative controls on original4TRAIN pairs.','Independent seven-file actual native backward/invalid-gradient/2step nonempty exact restore and matched inference/cost tests.','Five adversarial exact-native-decoder CPU small-support controls;1/8/128/406/512 valid supports paired with512.','Read-only scoped Git evidence and existing Y50-only authorization; no scoped Y54 commit or automatic admission.']
        work['design_cycle']=dict(stop_reason='external_resource_blocked',attempted_ids=['AdaRES-Y53','AdaRES-Y54'],evidence=[ref('pipeline/analyses/y53_padding_falsification_'+TOKEN+'.json'),ref(review_path)],resume_condition=review['required_human_action'],clarification='Bounded reserve now includes existing blockedY52 and prepared blockedY54. External dependency is explicit human scoped commit permission, NOT unavailable GPU or unsupported route-exhaustion. No executable unrun gate remains in these prepared candidates; Y52 is unchanged/not_due.')
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(review=ref(review_path),freeze=ref(freeze_path),stage=candidate['stage'],blocker_kind=candidate['blocker_kind'],retry_after=candidate['retry_after'],cost=costs,behavioral_parity_all8=True,evidence_count=len(candidate['evidence']),code_commit=None,admissions=0,training_launches=0),indent=2))
    finally:
        lock.close()


if __name__=='__main__':
    main()
