"""Separate observed CPU seed-skip feasibility from future native sparse TRAIN."""
import math
from y96_observer_core import check_response_rows


CID = "AdaRES-Y96"


def gradients(rows, cpu):
    assert len(rows) == 2
    for index, row in enumerate(rows):
        assert type(row["step"]) is int and row["step"] == index
        assert row["tensors"] > 0 and row["all_finite"] is True
        assert row["indicator_tensors"] == 10 and row["indicator_calls"] == row["prediction_calls"] == 7
        assert row["query_norm_consumption_calls"] == 7 and row["generator_layer_calls"] == 5
        assert row["initialization_calls"] == 1 and row["coarse_gate_consumers"] == 6
        assert row["both_input_gradients_finite"] is True
        for key in ("new_gradient_max", "parameter_update_max"):
            assert math.isfinite(row[key]) and row[key] > 0
        if cpu:
            assert math.isfinite(row["synthetic_CPU_decoder_loss"]) and row["decoder_input_gradients_finite"] is True


def validate_cpu(value):
    assert value["id"] == CID and value["parameters"] == 164695173 and value["additional_parameters"] == 65536
    for key in (
        "complete", "implementation_test_passed", "full_model_CPU_constructed", "configured_TRAIN_data_collated",
        "released_backbone_loaded", "native_YAML_semantically_unchanged", "train_eval_train_roundtrip",
        "decoder_initial_RNG_and_parent_weights_exact", "decoder_all_stages_anchor_exact",
        "decoder_input_gradients_anchor_close", "original_sampled_seed_and_generator_consumers_verified",
        "initial_and_first_coarse_consumers_verified", "static_mask_and_original_hard_gate_preserved",
        "original_shared_head_query_norm_consumption_verified", "nonzero_seed_skip_changes_initial_and_final_masks",
        "exact_bypass_restoration",
    ):
        assert value[key] is True, key
    assert value["points"] > 0 and value["expressions"] >= 2
    assert {row["dtype"] for row in value["component_controls"]} == {"torch.float32", "torch.float64"}
    assert all(flag is True for row in value["component_controls"] for key, flag in row.items() if key != "dtype")
    assert value["synthetic_CPU_decoder_optimizer_steps"] == 2
    gradients(value["CPU_decoder_steps"], True)
    check_response_rows(value["CPU_initialization_paths"])
    for key in ("CUDA_initialized", "native_smoke_passed", "native_task_backward",
                "native_sparse_forward", "validation_used", "admission"):
        assert value[key] is False


def validate_result(value, candidate, frozen):
    assert value["complete"] is True and value["id"] == candidate["id"] == frozen["variant"]["id"] == CID
    assert value["config"] == candidate["config"] == frozen["variant"]["config"]
    assert value["code_commit"] == candidate["code_commit"] == frozen["code_commit"]
    assert value["goal_sha256"] == frozen["goal_sha256"]
    assert value["native_YAML_sha256"] == frozen["variant"]["config_data"]["sha256"]
    assert value["native_entry"] == "tools/train.py::train" and value["split"] == "train"
    assert value["steps"] == value["optimizer_steps"] == 2
    assert len(value["losses"]) == 2 and all(math.isfinite(loss) for loss in value["losses"])
    for key in ("native_data_loaded", "backward_finite", "intended_path_observed",
                "static_mask_features_preserved", "original_hard_gate_algorithm_preserved", "native_modes_verified"):
        assert value[key] is True
    for key in ("validation_used", "model_weights_saved", "admission"):
        assert value[key] is False
    assert value["initialization_calls"] == 2 and value["generator_layer_calls"] == 10
    assert value["coarse_gate_consumers"] == 12
    assert value["shared_head_calls"] == value["query_norm_consumption_calls"] == 14
    gradients(value["gradient_checks"], False)
    check_response_rows(value["initialization_paths"])
    assert len(value["batches"]) == 2
    assert all(len(row["scan_ids"]) == 2 and row["expressions"] >= 2 and row["points"] > 0
               for row in value["batches"])
    assert len({scene for row in value["batches"] for scene in row["scan_ids"]}) == 4
    assert 0 < value["peak_reserved_bytes"] < 24 * 2**30
    power, data = {
        "server-1": (350., "/data1/songcl/Project/AdaRES_project/data"),
        "server-2": (280., "/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data"),
    }[frozen["runtime_server"]]
    assert frozen["runtime_physical_gpu"] == 0 and frozen["runtime_power_limit_watts"] == power
    assert value["data_source"]["config_data_root"].rstrip("/") == data
    assert value["data_source"]["dataset_class"] == "ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge"
