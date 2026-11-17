"""Validate CPU feasibility separately from the mandatory native TRAIN task."""
import math
from y100_observer_core import check_response_rows
CID = "AdaRES-Y100"


def gradients(rows, cpu):
    assert len(rows) == 2
    for i, r in enumerate(rows):
        assert type(r["step"]) is int and r["step"] == i
        assert r["tensors"] > 0 and r["all_finite"] is True
        assert r["indicator_tensors"] == 10 and r["indicator_calls"] == r["prediction_calls"] == 7
        assert r["query_norm_consumption_calls"] == 7 and r["coarse_gate_consumers"] == 6
        assert r["appearance_input_calls"] == r["original_appearance_projection_calls"] == r["original_spatial_projection_calls"] == r["original_base_mask_calls"] == r["original_sum_norm_calls"] == r["first_language_memory_consumers"] == 1
        assert r["original_visual_parameter_tensors"] == 14
        assert r["base_adapted_fusion_and_mask_gradients_finite"] is True and r["frozen_text_encoder_preserved"] is True
        assert type(r["raw_appearance_requires_grad"]) is bool
        for k in ("new_gradient_max", "parameter_update_max"):
            assert math.isfinite(r[k]) and r[k] > 0
        if cpu:
            assert math.isfinite(r["synthetic_CPU_decoder_loss"]) and r["decoder_input_gradients_finite"] is True


def validate_cpu(v):
    assert v["id"] == CID and v["parameters"] == 164891781 and v["additional_parameters"] == 262144
    for k in ("complete", "implementation_test_passed", "full_model_CPU_constructed", "configured_TRAIN_data_collated",
              "released_backbone_loaded", "native_YAML_semantically_unchanged", "train_eval_train_roundtrip",
              "decoder_initial_RNG_and_parent_weights_exact", "decoder_all_stages_anchor_exact",
              "decoder_input_gradients_anchor_close", "original_appearance_projection_and_fusion_consumers_verified",
              "all_seven_original_static_masks_verified", "static_mask_and_original_hard_gate_preserved",
              "original_shared_head_query_norm_consumption_verified", "nonzero_appearance_input_skip_changes_initial_and_final_masks",
              "exact_bypass_restoration"):
        assert v[k] is True, k
    assert v["points"] > 0 and v["expressions"] >= 2
    assert {r["dtype"] for r in v["component_controls"]} == {"torch.float32", "torch.float64"}
    assert all(flag is True for r in v["component_controls"] for k, flag in r.items() if k != "dtype")
    assert v["synthetic_CPU_decoder_optimizer_steps"] == 2
    gradients(v["CPU_decoder_steps"], True)
    check_response_rows(v["CPU_appearance_input_paths"])
    for k in ("CUDA_initialized", "native_smoke_passed", "native_task_backward", "native_sparse_forward", "validation_used", "admission"):
        assert v[k] is False


def validate_result(v, candidate, frozen):
    assert v["complete"] is True and v["id"] == candidate["id"] == frozen["variant"]["id"] == CID
    assert v["config"] == candidate["config"] == frozen["variant"]["config"]
    assert v["code_commit"] == candidate["code_commit"] == frozen["code_commit"]
    assert v["goal_sha256"] == frozen["goal_sha256"] and v["native_YAML_sha256"] == frozen["variant"]["config_data"]["sha256"]
    assert v["native_entry"] == "tools/train.py::train" and v["split"] == "train"
    assert v["steps"] == v["optimizer_steps"] == 2 and len(v["losses"]) == 2 and all(math.isfinite(n) for n in v["losses"])
    for k in ("native_data_loaded", "backward_finite", "intended_path_observed", "static_mask_features_preserved", "original_hard_gate_algorithm_preserved", "native_modes_verified"):
        assert v[k] is True, k
    for k in ("validation_used", "model_weights_saved", "admission"):
        assert v[k] is False
    assert v["appearance_input_calls"] == v["original_appearance_projection_calls"] == v["original_spatial_projection_calls"] == v["original_base_mask_calls"] == v["original_sum_norm_calls"] == v["first_language_memory_consumers"] == 2
    assert v["coarse_gate_consumers"] == 12 and v["shared_head_calls"] == v["query_norm_consumption_calls"] == 14
    gradients(v["gradient_checks"], False)
    check_response_rows(v["appearance_input_paths"])
    assert len(v["batches"]) == 2 and all(len(r["scan_ids"]) == 2 and r["expressions"] >= 2 and r["points"] > 0 for r in v["batches"])
    assert len({s for r in v["batches"] for s in r["scan_ids"]}) == 4
    assert 0 < v["peak_reserved_bytes"] < 24 * 2**30
    power, data = {"server-1": (350., "/data1/songcl/Project/AdaRES_project/data"), "server-2": (280., "/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data")}[frozen["runtime_server"]]
    assert frozen["runtime_physical_gpu"] == 0 and frozen["runtime_power_limit_watts"] == power
    assert v["data_source"]["config_data_root"].rstrip("/") == data
    assert v["data_source"]["dataset_class"] == "ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge"
