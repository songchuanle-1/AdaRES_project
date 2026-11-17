"""Separate actual CPU preparation from mandatory original sparse TRAIN evidence."""
import math


CID = "AdaRES-Y94"


def validate_cpu(cpu):
    assert cpu["id"] == CID and cpu["parameters"] == 164629661 and cpu["additional_parameters"] == 24
    for key in (
        "complete", "implementation_test_passed", "full_model_CPU_constructed",
        "configured_TRAIN_data_collated", "released_backbone_loaded", "native_YAML_semantically_unchanged",
        "train_eval_train_roundtrip", "actual_attention_zero_anchor_exact",
        "actual_hard_exclusions_preserved", "actual_attention_bias_formula_verified",
        "decoder_initial_RNG_and_parent_weights_exact", "decoder_all_stages_anchor_exact",
        "decoder_input_gradients_anchor_close", "decoder_backward_finite",
        "decoder_geometry_gradient_finite_nonzero", "nonzero_geometry_changes_final_masks",
        "initial_prediction_unchanged", "bypassed_geometry_negative_control_exact",
    ):
        assert cpu[key] is True, key
    assert cpu["points"] > 0 and cpu["expressions"] >= 2
    assert cpu["decoder_bias_calls"] == 1 and cpu["decoder_query_self_consumption_calls"] == 6
    assert cpu["decoder_indicator_calls"] == 7
    assert {row["dtype"] for row in cpu["component_controls"]} == {"torch.float32", "torch.float64"}
    assert all(value is True for row in cpu["component_controls"]
               for key, value in row.items() if key != "dtype")
    for key in ("CUDA_initialized", "native_smoke_passed", "native_task_backward",
                "native_sparse_forward", "validation_used", "admission"):
        assert cpu[key] is False
    assert cpu["synthetic_CPU_decoder_optimizer_steps"] == 2 and len(cpu["CPU_decoder_steps"]) == 2
    for index, row in enumerate(cpu["CPU_decoder_steps"]):
        assert type(row["step"]) is int and row["step"] == index
        assert row["all_finite"] is True and row["input_gradients_finite"] is True
        assert row["original_gradient_tensors"] > 0 and math.isfinite(row["loss"])
        for key in ("new_gradient_max", "parameter_update_max"):
            assert math.isfinite(row[key]) and row[key] > 0


def validate_result(raw, candidate, frozen):
    assert raw["complete"] is True and raw["id"] == candidate["id"] == frozen["variant"]["id"] == CID
    assert raw["config"] == candidate["config"] == frozen["variant"]["config"]
    assert raw["code_commit"] == candidate["code_commit"] == frozen["code_commit"]
    assert raw["goal_sha256"] == frozen["goal_sha256"]
    assert raw["native_YAML_sha256"] == frozen["variant"]["config_data"]["sha256"]
    assert raw["native_entry"] == "tools/train.py::train" and raw["split"] == "train"
    assert raw["steps"] == raw["optimizer_steps"] == 2
    assert len(raw["losses"]) == 2 and all(math.isfinite(value) for value in raw["losses"])
    for key in ("native_data_loaded", "backward_finite", "intended_path_observed"):
        assert raw[key] is True
    for key in ("validation_used", "model_weights_saved", "admission"):
        assert raw[key] is False
    stages = list(range(6))
    assert raw["geometry_calls"] == 2 and raw["shared_head_calls"] == 14
    assert raw["query_self_stage_calls"] == raw["attention_stage_calls"] == stages * 2
    assert len(raw["gradient_checks"]) == len(raw["geometry_paths"]) == 2
    for index, row in enumerate(raw["gradient_checks"]):
        assert row["tensors"] > 0 and row["all_finite"] is True
        assert row["indicator_tensors"] == 10 and row["indicator_calls"] == 7
        assert row["geometry_calls"] == 1 and row["attention_calls"] == 6
        assert row["query_self_stage_calls"] == stages
        assert row["native_hard_masks_preserved"] is True and row["query_gradients_finite"] is True
        for key in ("geometry_weight_gradient_max", "parameter_update_max"):
            assert math.isfinite(row[key]) and row[key] > 0
        path = raw["geometry_paths"][index]
        for key in ("native_positions_verified", "hard_masks_verified",
                    "consumed_by_all_query_self_attentions"):
            assert path[key] is True
        assert path["heads"] == 8 and path["queries"] == 128 and path["expressions"] >= 2
        assert path["reference_expression_rows"] == [0, path["expressions"] - 1]
        assert path["reference_head_rows"] == [0, 7] and path["reference_query_rows"] == [0, 127]
        assert type(path["weight_nonzero"]) is bool and path["weight_nonzero"] is bool(index)
        assert math.isfinite(path["reference_error_max"]) and path["reference_error_max"] >= 0
        assert math.isfinite(path["bias_max"]) and (path["bias_max"] > 0 if index else path["bias_max"] == 0)
    assert len(raw["batches"]) == 2
    assert all(len(row["scan_ids"]) == 2 and row["expressions"] >= 2 and row["points"] > 0
               for row in raw["batches"])
    assert len({scene for row in raw["batches"] for scene in row["scan_ids"]}) == 4
    assert 0 < raw["peak_reserved_bytes"] < 24 * 2**30 and frozen["runtime_physical_gpu"] == 0
    power, data = {
        "server-1": (350., "/data1/songcl/Project/AdaRES_project/data"),
        "server-2": (280., "/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data"),
    }[frozen["runtime_server"]]
    assert frozen["runtime_power_limit_watts"] == power
    assert raw["data_source"]["config_data_root"].rstrip("/") == data
    assert raw["data_source"]["dataset_class"] == (
        "ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge")
