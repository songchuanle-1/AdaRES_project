"""Read exact terminal checkpoints and source bindings; never resume training."""

import hashlib
import json
from pathlib import Path
import subprocess

import torch

assert P["physical_gpu"] is None
root = Path(P["root"])
assert subprocess.check_output(["hostname"]).decode().strip() == "dl01-MS-7C60"
assert (root / ".codex_source_commit").read_text().strip() == BINDINGS["commit"]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


for source in BINDINGS["sources"]:
    raw = (root / source["path"]).read_bytes()
    assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == source["lf_sha256"], source["path"]
commands = []
for command_path in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        argv = command_path.read_bytes().split(b"\0")
    except OSError:
        continue
    if b"tools/train.py" in argv:
        commands.append(argv)
methods = {}
for identity, binding in BINDINGS["methods"].items():
    assert not any(binding["config"].encode() in argv for argv in commands), identity
    assert digest(root / binding["config"]) == binding["config_sha256"]
    path = Path(binding["checkpoint"])
    assert path.is_file()
    sha = digest(path)
    if binding.get("checkpoint_sha256"):
        assert sha == binding["checkpoint_sha256"], identity
    checkpoint = torch.load(str(path), map_location="cpu")
    meta = checkpoint["meta"]
    observation = dict(
        path=str(path), sha256=sha, bytes=path.stat().st_size,
        epoch=int(meta["epoch"]), best_miou=float(meta["best_miou"]),
        model_tensors=len(checkpoint["model"]))
    del checkpoint
    row = dict(binding, checkpoint_sha256=sha, best=observation)
    if "last" in binding:
        last = Path(binding["last"])
        checkpoint = torch.load(str(last), map_location="cpu")
        meta = checkpoint["meta"]
        assert int(meta["epoch"]) == 30, "Fixed30 terminal checkpoint missing"
        optimizer = checkpoint["optimizer"]
        scheduler = checkpoint["lr_scheduler"]
        row["last_observation"] = dict(
            path=str(last), sha256=digest(last), bytes=last.stat().st_size,
            epoch=int(meta["epoch"]), best_miou=float(meta["best_miou"]),
            optimizer_state_tensors=len(optimizer["state"]),
            scheduler_last_epoch=int(scheduler["last_epoch"]))
        assert optimizer["state"] and int(scheduler["last_epoch"]) == 30
        assert float(meta["best_miou"]) == observation["best_miou"]
        del checkpoint
        row["fixed30_finished"] = True
    methods[identity] = row
assert (root / ".codex_source_commit").read_text().strip() == BINDINGS["commit"]
result = dict(
    complete=True, token=P["token"], code_commit=BINDINGS["commit"],
    methods=methods, source_bindings=BINDINGS["sources"],
    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    mount=subprocess.check_output(
        ["findmnt", "-n", "-o", "SOURCE,TARGET,UUID", "--target", "/bak_home"]).decode().strip(),
    no_training_process_for_methods=True, no_checkpoint_or_source_writes=True,
    limitation="Checkpoint metadata is not a fresh full evaluation. Monitor segment best "
               "is not substituted for carried canonical best. No training/restart/admission.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
