"""Additional native-only guards without rewriting the preserved CPU contract."""
from y100_contract import validate_result as original_validate_result


def validate_result(value, candidate, frozen):
    original_validate_result(value, candidate, frozen)
    assert all(r["raw_appearance_requires_grad"] is False for r in value["gradient_checks"])
    assert all(r["raw_appearance_requires_grad"] is False for r in value["appearance_input_paths"])
