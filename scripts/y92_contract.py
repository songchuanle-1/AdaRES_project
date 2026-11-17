"""Validate actual visual-side quadratic evidence, keeping CPU and native distinct."""
import math
from y92_observer_core import check_response_rows

CID = 'AdaRES-Y92'


def validate_gradients(rows):
    assert len(rows) == 2
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index
        assert row['all_finite'] is True and row['input_gradients_finite'] is True
        assert row['tensors'] > 0 and row['indicator_tensors'] == 10 and row['projection_calls'] == 1
        for key in ('original_mask_consumers', 'original_score_consumers',
                    'indicator_calls', 'query_norm_consumption_calls'):
            assert row[key] == 7
        assert row['coarse_gate_consumers'] == 6
        for key in ('new_gradient_max', 'parameter_update_max'):
            assert math.isfinite(row[key]) and row[key] > 0


def validate_cpu(cpu):
    assert cpu['id'] == CID and cpu['parameters'] == 164695173 and cpu['additional_parameters'] == 65536
    for key in ('complete', 'implementation_test_passed', 'full_model_CPU_constructed',
        'configured_TRAIN_data_collated', 'released_backbone_loaded', 'native_YAML_semantically_unchanged',
        'train_eval_train_roundtrip', 'decoder_initial_RNG_and_parent_weights_exact',
        'decoder_all_stages_anchor_exact', 'decoder_input_gradients_anchor_close',
        'intended_initial_masks_changed_initial_scores_unchanged', 'bypass_restoration_exact',
        'original_shared_head_query_norm_consumption_verified',
        'shared_visual_features_and_original_hard_gate_preserved',
        'original_score_and_mask_consumers_verified'):
        assert cpu[key] is True, key
    assert cpu['points'] > 0 and cpu['expressions'] >= 2
    assert cpu['synthetic_CPU_decoder_optimizer_steps'] == 2
    assert all(cpu[key] is False for key in ('CUDA_initialized', 'native_smoke_passed',
        'native_task_backward', 'native_sparse_forward', 'validation_used', 'admission'))
    assert {row['dtype'] for row in cpu['component_controls']} == {'torch.float32', 'torch.float64'}
    assert all(value is True for row in cpu['component_controls'] for key, value in row.items() if key != 'dtype')
    check_response_rows(cpu['CPU_projection_paths'])
    validate_gradients(cpu['CPU_decoder_steps'])


def validate_result(raw, candidate, frozen):
    assert raw['complete'] is True and raw['id'] == candidate['id'] == frozen['variant']['id'] == CID
    assert raw['config'] == candidate['config'] == frozen['variant']['config']
    assert raw['code_commit'] == candidate['code_commit'] == frozen['code_commit']
    assert raw['goal_sha256'] == frozen['goal_sha256']
    assert raw['native_YAML_sha256'] == frozen['variant']['config_data']['sha256']
    assert raw['steps'] == raw['optimizer_steps'] == 2
    assert len(raw['losses']) == 2 and all(math.isfinite(value) for value in raw['losses'])
    assert raw['native_entry'] == 'tools/train.py::train' and raw['split'] == 'train'
    for key in ('native_data_loaded', 'backward_finite', 'intended_path_observed', 'native_modes_verified',
                'shared_transformed_visual_mask_features_consumed', 'original_hard_gate_algorithm_preserved'):
        assert raw[key] is True
    assert raw['validation_used'] is False and raw['model_weights_saved'] is False and raw['admission'] is False
    assert raw['projection_calls'] == 2 and raw['coarse_gate_consumers'] == 12
    for key in ('original_mask_consumers', 'original_score_consumers', 'shared_head_calls',
                'query_norm_consumption_calls'):
        assert raw[key] == 14
    validate_gradients(raw['gradient_checks'])
    check_response_rows(raw['projection_paths'])
    assert len(raw['batches']) == 2
    assert all(len(batch['scan_ids']) == 2 and batch['expressions'] > 0 and batch['points'] > 0
               for batch in raw['batches'])
    assert len({scene for batch in raw['batches'] for scene in batch['scan_ids']}) == 4
    assert 0 < raw['peak_reserved_bytes'] < 24 * 2**30 and frozen['runtime_physical_gpu'] == 0
    power, data = {'server-1': (350., '/data1/songcl/Project/AdaRES_project/data'),
                   'server-2': (280., '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data')}[frozen['runtime_server']]
    assert frozen['runtime_power_limit_watts'] == power
    assert raw['data_source']['dataset_class'] == 'ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'
    assert raw['data_source']['config_data_root'].rstrip('/') == data
