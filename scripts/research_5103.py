"""Scoped utilities for one owned AdaRES research continuation."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "5103a354430e47b3b7a04162dcaae8af"
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import foreground_tick as tick
import pipeline_monitor as monitor
import closure


def lease(reserve=90):
    owner = tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")
    assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    remaining = (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds()
    assert remaining > reserve
    for name, key in ((".tick_context.json", "context_sha256"),
                      (".tick_context_state.next.json", "pending_sha256")):
        assert tick.digest(ROOT / "pipeline/headless" / name) == owner[key]
    return owner, remaining


def ref(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    assert path.resolve().is_relative_to(ROOT.resolve()) and path.is_file()
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, value):
    assert not (ROOT / path).exists(), "Never overwrite frozen evidence"
    tick.save(ROOT / path, value)
    return ref(path)


def bind(candidate, *rows):
    for row in rows:
        if row not in candidate["evidence"]:
            candidate["evidence"].append(row)
