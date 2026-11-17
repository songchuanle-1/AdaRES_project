"""Validate full-scene geometry CPU feasibility separately from native TRAIN."""
import math
from y102_observer_core import check_response_rows
CID='AdaRES-Y102'

def gradients(rows,cpu):
    assert len(rows)==2
    for i,r in enumerate(rows):
        assert type(r['step']) is int and r['step']==i and r['all_finite'] is True and r['tensors']>0
        assert r['indicator_tensors']==10 and r['indicator_calls']==r['prediction_calls']==r['query_norm_consumption_calls']==7
        assert r['coarse_gate_consumers']==6 and r['geometry_mask_calls']==1 and r['original_visual_parameter_tensors']==14
        assert r['original_base_adapted_and_visual_gradients_finite'] is True and r['frozen_text_encoder_preserved'] is True
        for k in ('new_gradient_max','parameter_update_max'):assert math.isfinite(r[k]) and r[k]>0
        if cpu:assert math.isfinite(r['synthetic_CPU_decoder_loss']) and r['decoder_input_gradients_finite'] is True

def validate_cpu(v):
    assert v['id']==CID and v['parameters']==164630405 and v['additional_parameters']==768
    for k in ('complete','implementation_test_passed','full_model_CPU_constructed','configured_TRAIN_data_collated','released_backbone_loaded',
              'native_YAML_semantically_unchanged','train_eval_train_roundtrip','decoder_initial_RNG_and_parent_weights_exact',
              'decoder_all_stages_anchor_exact','decoder_input_gradients_anchor_close','original_geometry_mass_and_static_mask_consumers_verified',
              'all_seven_static_mask_consumers_verified','static_mask_and_original_hard_gate_preserved','original_shared_head_query_norm_consumption_verified',
              'nonzero_geometry_skip_changes_initial_and_final_masks','exact_bypass_restoration'):assert v[k] is True,k
    assert v['points']>0 and v['expressions']>=2
    assert {r['dtype'] for r in v['component_controls']}=={'torch.float32','torch.float64'}
    assert all(x is True for r in v['component_controls'] for k,x in r.items() if k!='dtype')
    assert v['synthetic_CPU_decoder_optimizer_steps']==2;gradients(v['CPU_decoder_steps'],True);check_response_rows(v['CPU_geometry_mask_paths'])
    for k in ('CUDA_initialized','native_smoke_passed','native_task_backward','native_sparse_forward','validation_used','admission'):assert v[k] is False

def validate_result(v,c,f):
    assert v['complete'] is True and v['id']==c['id']==f['variant']['id']==CID
    assert v['config']==c['config']==f['variant']['config'] and v['code_commit']==c['code_commit']==f['code_commit']
    assert v['goal_sha256']==f['goal_sha256'] and v['native_YAML_sha256']==f['variant']['config_data']['sha256']
    assert v['native_entry']=='tools/train.py::train' and v['split']=='train'
    assert v['steps']==v['optimizer_steps']==2 and len(v['losses'])==2 and all(math.isfinite(x) for x in v['losses'])
    for k in ('native_data_loaded','backward_finite','intended_path_observed','original_base_mask_and_static_reuse_preserved','original_hard_gate_algorithm_preserved','native_modes_verified'):assert v[k] is True,k
    for k in ('validation_used','model_weights_saved','admission'):assert v[k] is False
    assert v['geometry_mask_calls']==v['original_visual_projection_and_mask_calls']==2 and v['coarse_gate_consumers']==12
    assert v['shared_head_calls']==v['query_norm_consumption_calls']==14
    gradients(v['gradient_checks'],False);check_response_rows(v['geometry_mask_paths'])
    assert len(v['batches'])==2 and all(len(r['scan_ids'])==2 and r['expressions']>=2 and r['points']>0 for r in v['batches'])
    assert len({s for r in v['batches'] for s in r['scan_ids']})==4 and 0<v['peak_reserved_bytes']<24*2**30
    power,data={'server-1':(350.,'/data1/songcl/Project/AdaRES_project/data'),'server-2':(280.,'/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data')}[f['runtime_server']]
    assert f['runtime_physical_gpu']==0 and f['runtime_power_limit_watts']==power
    assert v['data_source']['config_data_root'].rstrip('/')==data and v['data_source']['dataset_class']=='ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'
