"""Bind equal YAML values before recording the actual remote config byte hash."""

import argparse
import ast
from datetime import datetime

import yaml

import authorize_prepared_candidates as scoped

ROOT = scoped.ROOT
TOKEN = "6debf88fed634a13b772f27bdc244bbd"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    assert args.token == TOKEN
    scoped.lease(args.token)
    lock = scoped.monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        prior = "pipeline/analyses/terminal_bindings_v2_" + TOKEN + "_program.py"
        tree = ast.parse((ROOT / prior).read_text(encoding="utf-8"))
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "BINDINGS"
                                  for t in n.targets))
        bindings = ast.literal_eval(assignment.value)
        for value in bindings["methods"].values():
            value["config_values"] = yaml.safe_load(
                (ROOT / "AdaRES" / value["config"]).read_text(encoding="utf-8"))
        assignment.value = ast.parse(repr(bindings), mode="eval").body
        main_loop = next(n for n in tree.body if isinstance(n, ast.For)
                         and isinstance(n.target, ast.Tuple)
                         and any(isinstance(t, ast.Name) and t.id == "identity"
                                 for t in n.target.elts))
        assertions = [(i, n) for i, n in enumerate(main_loop.body) if isinstance(n, ast.Assert)
                      and "binding['config_sha256']" in ast.unparse(n.test)]
        assert len(assertions) == 1
        index, _ = assertions[0]
        main_loop.body[index:index + 1] = ast.parse("""
assert yaml.safe_load((root / binding['config']).read_text()) == binding['config_values'], identity
binding = dict(binding, config_sha256=digest(root / binding['config']))
""").body
        tree.body.insert(1, ast.Import(names=[ast.alias(name="yaml")]))
        program = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        ast.parse(program, feature_version=(3, 8))
        path = "pipeline/analyses/terminal_bindings_v3_" + TOKEN + "_program.py"
        assert not (ROOT / path).exists()
        (ROOT / path).write_text(program, encoding="utf-8", newline="\n")
        plan_path = path.replace("_program.py", "_prereg.json")
        plan = dict(
            complete=True, token=TOKEN, preregistered_at=datetime.now().astimezone().isoformat(),
            previous_program=scoped.ref(prior),
            previous_failed_log=scoped.ref("pipeline/analyses/terminal_bindings_v2_" + TOKEN + ".log"),
            program=scoped.ref(path),
            scope="Config binding serialization compatibility only, no scientific or runtime edits.",
            change="Require ALL parsed YAML values equal the original local bound config, "
                   "then record full actual remote config SHA for unchanged evaluator binding. "
                   "No path/config semantic exceptions and no overwriting runtime files.",
            reason="Local and deployed YAML byte hashes differ; equal parsed values must "
                   "be demonstrated rather than asserted or copying one over the other.",
            result="pending")
        scoped.tick.save(ROOT / plan_path, plan)
        work = scoped.tick.load(ROOT / "pipeline/research_work.json")
        for candidate in work["candidates"]:
            if candidate["id"] in ("AdaRES-Y52", "AdaRES-Y54"):
                candidate["evidence"].extend([scoped.ref(path), scoped.ref(plan_path),
                    plan["previous_failed_log"]])
        scoped.lease(args.token)
        scoped.tick.save(ROOT / "pipeline/research_work.json", work)
        print("Froze complete YAML-value equality check plus actual remote byte binding.")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
