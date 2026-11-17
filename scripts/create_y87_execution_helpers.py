"""Bind the unchanged tested Y87 model directly to the deployed Y85 parent."""
import argparse
import ast
from pathlib import Path
from create_y85_execution_helpers import replace_function
from prepare_y60_stage_heads import ROOT, lease, monitor, tick

SCRIPTS = Path(__file__).resolve().parent

VALIDATE = '''def validate_result(raw, candidate, frozen):
    assert raw['complete'] is True and raw['id'] == candidate['id'] == CID == frozen['variant']['id']
    assert raw['config'] == candidate['config'] == frozen['variant']['config']
    assert raw['code_commit'] == frozen['code_commit'] == Y87
    assert raw['goal_sha256'] == frozen['goal_sha256']
    assert raw['native_YAML_sha256'] == frozen['variant']['config_data']['sha256']
    assert raw['steps'] == raw['optimizer_steps'] == 2
    assert len(raw['losses']) == 2 and all(math.isfinite(value) for value in raw['losses'])
    assert raw['native_entry'] == 'tools/train.py::train' and raw['split'] == 'train'
    for key in ('native_data_loaded', 'backward_finite', 'intended_path_observed',
                'native_modes_verified', 'static_mask_features_preserved',
                'original_hard_gate_algorithm_preserved', 'original_initial_prediction_preserved'):
        assert raw[key] is True
    assert raw['validation_used'] is False and raw['model_weights_saved'] is False
    assert raw['shared_head_calls'] == raw['query_norm_consumption_calls'] == 14
    assert raw['fusion_calls'] == raw['original_norm_consumption_calls'] == raw['original_ffn_consumption_calls'] == 12
    assert len(raw['gradient_checks']) == len(raw['batches']) == 2
    from test_y87_native_plan import N
    N['check_response_rows'](raw['fusion_paths'])
    for index, row in enumerate(raw['gradient_checks']):
        assert type(row['step']) is int and row['step'] == index
        assert row['all_finite'] is True and row['tensors'] > 0 and row['indicator_tensors'] == 10
        assert row['indicator_calls'] == row['query_norm_consumption_calls'] == 7
        assert row['fusion_calls'] == row['original_norm_consumption_calls'] == row['original_ffn_consumption_calls'] == 6
        assert row['both_input_gradients_finite'] is True
        assert all(math.isfinite(row[key]) and row[key] > 0 for key in ('new_gradient_max', 'parameter_update_max'))
    assert 0 < raw['peak_reserved_bytes'] < 24 * 2**30
    assert all(len(batch['scan_ids']) == 2 and batch['expressions'] > 0 and batch['points'] > 0
               for batch in raw['batches'])
    assert len({scene for batch in raw['batches'] for scene in batch['scan_ids']}) == 4
    assert raw['data_source']['dataset_class'] == 'ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'
    assert raw['data_source']['config_data_root'].rstrip('/') == '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data'
'''


def create(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y87')
        assert candidate['stage'] == 'validation' and candidate['code_commit'] == '47ec0813cb4131ddf3e0a08db4fa9ac877123685'
        outputs = {}
        for filename in ('y85_deployment.py', 'advance_y85.py'):
            text = (SCRIPTS / filename).read_text(encoding='utf-8')
            text = text.replace('Y85', 'Y87').replace('y85', 'y87')
            text = text.replace('Y84', 'Y85').replace('y84', 'y85')
            text = text.replace('aa9d8db3ea05432181ada752eaeeca27952aae27', candidate['code_commit'])
            if filename == 'y85_deployment.py':
                text = text.replace('Y85 = previous.Y85', 'Y85 = previous.Y85\nY84 = previous.Y84')
                text = text.replace('adares_y87_mask_feature_norm', 'adares_y87_self_language_product')
                old = "ADDITIONS = dict(zip(VERSIONS[1:], ('AdaRES-Y' + str(n) for n in range(72, 86))))"
                assert old in text
                text = text.replace(old, "ADDITIONS = dict(previous.ADDITIONS, **{Y87: 'AdaRES-Y87'})")
                text = text.replace('previous.previous.NAMES', 'previous.previous.previous.NAMES')
                text = text.replace("('AdaRES-Y85', 'AdaRES-Y87')", "('AdaRES-Y84', 'AdaRES-Y87')")
                text = text.replace("{'AdaRES-Y85': ('server-1', Y85, 350.),",
                                    "{'AdaRES-Y84': ('server-1', Y84, 350.),")
            else:
                text = text.replace("FROZEN_GRANT = 'pipeline/authorizations/server2_gpu1_20261029.json'",
                                    'FROZEN_GRANT = GRANT')
                text = replace_function(text, 'validate_result', VALIDATE)
                text = text.replace("'AdaRES-Y83'", "'AdaRES-Y85'")
                text = text.replace('Y83_terminal', 'Y85_terminal')
                text = text.replace('shared visual mask-feature normalization',
                                    'self-language product before the original shared norm and FFN')
                text = text.replace('shared mask normalization paths verified. Active output equivalence to W2 is not claimed.',
                                    'self-language product paths, original norm/FFN consumers and two optimizer updates verified.')
                text = text.replace('Preserve Y85 and both user pauses.', 'Preserve Y84 and both user pauses.')
            ast.parse(text, feature_version=(3, 8))
            outputs[SCRIPTS / filename.replace('y85', 'y87')] = text
        assert all(not path.exists() for path in outputs)
        for path, text in outputs.items():
            path.write_text(text, encoding='utf-8', newline='\n')
        print('Created Y87 lifecycle helpers; only its seven tested files may be selected.')
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    create(parser.parse_args().token)
