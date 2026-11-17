"""Keep current-query FFN CPU feasibility distinct from required native TRAIN."""
import math
from y101_observer_core import check_response_rows
CID="AdaRES-Y101"


def gradients(rows,cpu):
    assert len(rows)==2
    for i,r in enumerate(rows):
        assert type(r['step']) is int and r['step']==i
        assert r['all_finite'] is True and r['tensors']>0
        assert r['indicator_tensors']==10 and r['indicator_calls']==r['prediction_calls']==r['query_norm_consumption_calls']==7
        assert r['coarse_gate_consumers']==r['query_FFN_skip_calls']==r['original_query_FFN_calls']==6
        assert r['original_query_FFN_parameter_tensors']==36
        for k in ('full_input_base_adapted_and_original_FFN_gradients_finite','original_shared_fusion_norm_gradients_finite','static_mask_gradients_finite','frozen_text_encoder_preserved'):
            assert r[k] is True,k
        for k in ('new_gradient_max','parameter_update_max'):
            assert math.isfinite(r[k]) and r[k]>0
        if cpu:
            assert math.isfinite(r['synthetic_CPU_decoder_loss']) and r['decoder_input_gradients_finite'] is True


def validate_cpu(v):
    assert v['id']==CID and v['parameters']==164695173 and v['additional_parameters']==65536
    for k in ('complete','implementation_test_passed','full_model_CPU_constructed','configured_TRAIN_data_collated',
              'released_backbone_loaded','native_YAML_semantically_unchanged','train_eval_train_roundtrip',
              'decoder_initial_RNG_and_parent_weights_exact','decoder_all_stages_anchor_exact','decoder_input_gradients_anchor_close',
              'all_six_complete_original_query_FFNs_verified','all_six_shared_current_query_bypasses_verified',
              'static_mask_and_original_hard_gate_preserved','original_shared_head_query_norm_consumption_verified',
              'initial_readout_exact_and_refined_masks_change','exact_bypass_restoration'):
        assert v[k] is True,k
    assert v['points']>0 and v['expressions']>=2
    assert {r['dtype'] for r in v['component_controls']}=={'torch.float32','torch.float64'}
    assert all(flag is True for r in v['component_controls'] for k,flag in r.items() if k!='dtype')
    assert v['synthetic_CPU_decoder_optimizer_steps']==2
    gradients(v['CPU_decoder_steps'],True)
    check_response_rows(v['CPU_query_FFN_paths'])
    for k in ('CUDA_initialized','native_smoke_passed','native_task_backward','native_sparse_forward','validation_used','admission'):
        assert v[k] is False


def validate_result(v,c,f):
    assert v['complete'] is True and v['id']==c['id']==f['variant']['id']==CID
    assert v['config']==c['config']==f['variant']['config'] and v['code_commit']==c['code_commit']==f['code_commit']
    assert v['goal_sha256']==f['goal_sha256'] and v['native_YAML_sha256']==f['variant']['config_data']['sha256']
    assert v['native_entry']=='tools/train.py::train' and v['split']=='train'
    assert v['steps']==v['optimizer_steps']==2 and len(v['losses'])==2 and all(math.isfinite(x) for x in v['losses'])
    for k in ('native_data_loaded','backward_finite','intended_path_observed','static_mask_features_preserved','original_hard_gate_algorithm_preserved','native_modes_verified'):
        assert v[k] is True,k
    for k in ('validation_used','model_weights_saved','admission'):
        assert v[k] is False
    assert v['query_FFN_skip_calls']==v['original_query_FFN_calls']==v['coarse_gate_consumers']==12
    assert v['shared_head_calls']==v['query_norm_consumption_calls']==14
    gradients(v['gradient_checks'],False);check_response_rows(v['query_FFN_paths'])
    assert len(v['batches'])==2 and all(len(r['scan_ids'])==2 and r['expressions']>=2 and r['points']>0 for r in v['batches'])
    assert len({s for r in v['batches'] for s in r['scan_ids']})==4
    assert 0<v['peak_reserved_bytes']<24*2**30
    power,data={'server-1':(350.,'/data1/songcl/Project/AdaRES_project/data'),'server-2':(280.,'/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data')}[f['runtime_server']]
    assert f['runtime_physical_gpu']==0 and f['runtime_power_limit_watts']==power
    assert v['data_source']['config_data_root'].rstrip('/')==data
    assert v['data_source']['dataset_class']=='ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'
