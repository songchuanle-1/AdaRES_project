# Y44 Native Sort Adapter Repair

First probe y44_native_controls_ff34 failed in captured_extract at
torch.argsort(sp, stable=True): server2's native PyTorch does not accept the
stable keyword. The failure preceded native decoded/control observations;
no valid result JSON and no gate PASS. Raw .log and original probe preserved.

V2 only replaces stable label sort by unique integer key
sp.long()*(point_count+1)+original_point_index. This produces exactly the
preregistered label-major/original-point order and withinSP reverse
permutation, using supported native argsort. No candidate model/source,
control/scene list/gate/tolerance/optimizer/resource change. The production
code package is unchanged. Version2 runs under a new result name.
