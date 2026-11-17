# Evaluation Serialization Failure

Tick3f09f39c723e4389b4b51274b1cd7aba, 2026-10-01.
The first native best-checkpoint val pass completed all363chunks on GPU1.
At first-pass persistence, json.dumps rejected checkpoint meta.best_miou
with TypeError: Object of type float32 is not JSON serializable.
SSH command exited1. No output JSON was written and no pass is claimed
as durable evaluation evidence. Training and checkpoints were not modified.

Repair: explicitly parse native epoch as int and best_miou as float before
serialization. No inference, configuration, ordering, metric or threshold
change. The repaired rerun uses a new unique output path and repeats the
entire full validation capture. Preserve this failure independently.

Separate B1 reference attempt: model import exited1 before inference because
root ipdn imports B3, whose Criterion conflicts with B1's unforced registry
decorator. No B1 output was written. Repair is confined to the standalone
evaluation process: verify the inherited Criterion comes from B3 and remove
that one entry before importing B1. B1 constructs its own original Criterion
directly. No training source, weights, loss or prediction semantics changed.
