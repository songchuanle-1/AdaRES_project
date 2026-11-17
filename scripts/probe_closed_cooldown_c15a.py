"""Reproduce the shared cooldown predicate without invoking or modifying it."""
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "c15addc5f96f406593c63f1dd9aa11f0"
SKILL = Path("C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
OUTPUT = ROOT / ("pipeline/analyses/closed_stage_cooldown_regression_" + TOKEN + ".json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    owner = load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")
    assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 180
    for name, key in ((".tick_context.json", "context_sha256"), (".tick_context_state.next.json", "pending_sha256")):
        assert digest(ROOT / "pipeline/headless" / name) == owner[key]
    ledger_path = ROOT / "pipeline/research_work.json"
    before_ledger = digest(ledger_path)
    work = load(ledger_path)
    view = load(ROOT / "pipeline/headless/.tick_model_view.json")
    assert view["research"]["due_count"] == 0 and view["research"]["reserve_count"] == 2
    assert view["research"]["reserve_budget"] == 0 and view["design_bootstrap"]["required"]
    source_path = SKILL / "foreground_tick.py"
    source_hash = digest(source_path)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    complete = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "complete")
    assignments = [node for node in ast.walk(complete) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pending_work" for target in node.targets)]
    assert len(assignments) == 1
    pending_expression = assignments[0].value
    conditions = [node.test for node in ast.walk(complete) if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And)
        and isinstance(node.test.values[0], ast.Name) and node.test.values[0].id == "pending_work"]
    assert len(conditions) == 1
    condition = conditions[0]

    def evaluate(candidates):
        namespace = dict(candidates=copy.deepcopy(candidates))
        expression = ast.fix_missing_locations(ast.Expression(copy.deepcopy(pending_expression)))
        pending = eval(compile(expression, str(source_path), "eval"), {}, namespace)
        namespace["pending_work"] = pending
        guard = ast.fix_missing_locations(ast.Expression(copy.deepcopy(condition)))
        decision = bool(eval(compile(guard, str(source_path), "eval"), {}, namespace))
        return dict(pending_count=len(pending), pending_stages=dict(Counter(row["stage"] for row in pending)),
            enters_cooldown_write_branch=decision)

    candidates = work["candidates"]
    terminal = {"training", "complete", "rejected", "closed"}
    true_pending = [row for row in candidates if row["stage"] not in terminal]
    closed = [row for row in candidates if row["stage"] == "closed"]
    assert len(true_pending) == 2 and {row["stage"] for row in true_pending} == {"blocked"}
    assert len(closed) == 12
    assert {row["blocker_kind"] for row in true_pending} == {"human_authorization"}
    retry_dates = [datetime.fromisoformat(row["retry_after"]) for row in true_pending]
    assert all(value > datetime.now(timezone.utc) for value in retry_dates)
    actual = evaluate(candidates)
    without_closed = evaluate([row for row in candidates if row["stage"] != "closed"])
    minimal = evaluate(true_pending)
    closed_only = evaluate(closed)
    assert actual["enters_cooldown_write_branch"] is False
    assert without_closed["enters_cooldown_write_branch"] is True
    assert minimal["enters_cooldown_write_branch"] is True
    assert closed_only["enters_cooldown_write_branch"] is False
    assert actual["pending_stages"] == {"closed": 12, "blocked": 2}
    cooldown_path = ROOT / "pipeline/headless/.research_cooldown.json"
    cooldown = load(cooldown_path)
    assert datetime.fromisoformat(cooldown["retry_after"]) < datetime.now(timezone.utc)
    assert digest(ledger_path) == before_ledger and digest(source_path) == source_hash
    assert not OUTPUT.exists()
    evidence = dict(schema_version=1, complete=True, token=TOKEN,
        kind="operational_counterexample_not_scientific_candidate",
        tested_at=datetime.now().astimezone().isoformat(),
        snapshot_at=view["generated_at"], source=dict(path=str(source_path), sha256=source_hash,
            predicate_line=assignments[0].lineno, cooldown_guard_line=condition.lineno),
        probe=dict(path="scripts/probe_closed_cooldown_c15a.py", sha256=digest(Path(__file__))),
        ledger_before_and_after_sha256=before_ledger,
        actual_ledger_predicate=actual, counterfactual_remove_closed_in_memory_only=without_closed,
        matched_blocked_only_control=minimal, closed_only_control=closed_only,
        actual_pending_ids=[row["id"] for row in true_pending],
        actual_pending_retry_after={row["id"]: row["retry_after"] for row in true_pending},
        expired_cooldown=dict(token=cooldown["token"], retry_after=cooldown["retry_after"]),
        prior_validated_external_stop=work["design_cycle"]["stop_reason"],
        conclusion="The unchanged shared predicate includes terminal legacy closed rows in pending_work. On this real ledger it bypasses external-block cooldown; removing only closed rows in memory activates the same guard. This explains why the prior externally blocked tick did not save a fresh cooldown. It does not establish a model defect or authorize runtime/policy changes.",
        unchanged_ledger=True, unchanged_shared_runtime=True, no_scheduler_changes=True,
        new_candidates=0, scientific_tests=0, model_changes=0, admissions=0, training_launches=0,
        limitations="Isolated source-AST predicate execution, not shared-code repair, not a new mechanism or scientific progress. No actual cooldown file was written or modified.")
    OUTPUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(evidence=str(OUTPUT), evidence_sha256=digest(OUTPUT),
        actual=actual, in_memory_without_closed=without_closed,
        source_unchanged=True, ledger_unchanged=True, scientific_candidates=0), indent=2))


if __name__ == "__main__":
    main()
