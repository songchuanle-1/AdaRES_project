"""Complete the unchanged five-method frozen comparison; preserve old partial review."""

import argparse
from datetime import datetime
import json
import statistics

from research_5103 import ROOT, TOKEN, lease, ref, save, bind, tick, monitor
from analyze_y27_frozen_terminal import comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    assert args.token == TOKEN
    lease()
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        paths = {name: "pipeline/analyses/frozen_" + name + "_6debf88fed634a13b772f27bdc244bbd.json"
                 for name in ("y52", "y54", "w2", "b1")}
        paths["y36"] = "pipeline/analyses/frozen_y36_" + TOKEN + ".json"
        captures = {name: tick.load(ROOT / path)["rows"] for name, path in paths.items()}
        for capture in captures.values():
            assert capture["complete"] and capture["completed_passes"] == 2
            assert all(p["summary"]["expressions"] == 9508 and p["summary"]["scenes"] == 141
                       for p in capture["passes"])
        differences = {name: {other: comparison(captures[name], captures[other])
                             for other in ("w2", "b1", "y36")} for name in ("y52", "y54")}
        harms = {name: [dict(reference=other, group=group, result=row)
                       for other, groups in results.items() for group, row in groups.items()
                       if group != "all" and row.get("scene_clustered_ci95", [0, 0])[1] < 0]
                 for name, results in differences.items()}
        path = "pipeline/analyses/final_terminal_comparison_" + TOKEN + ".json"
        result = dict(complete=True, token=TOKEN, observed_at=datetime.now().astimezone().isoformat(),
            source_captures={name: ref(path) for name, path in paths.items()},
            prior_partial_review=ref("pipeline/analyses/design_and_terminal_review_6debf88fed634a13b772f27bdc244bbd.json"),
            primary="FIRST, not better repeat", bootstrap=dict(unit="scene", draws=10000, seed=1999),
            size_cuts=[1487, 3325], summary={name: c["passes"][0]["summary"] for name, c in captures.items()},
            comparisons=differences, frozen_material_harm_review=harms,
            costs={name: dict(parameters=c["parameters"], warm_first_chunk_median_seconds=statistics.median(
                c["warm_first_chunk_seconds"]), peak_allocated_gib=c["peak_allocated_gib"],
                peak_reserved_gib=c["peak_reserved_gib"], repeat_miou_difference=c["repeat_miou_difference"],
                repeat_max_expression_iou_difference=c["repeat_max_iou_difference"])
                   for name, c in captures.items()},
            decision="Neither Y52 nor Y54 reaches goal or exceeds FIRST champion mean; "
                     "no automatic promotion/rerun or universal mechanism rejection. "
                     "Preserve all paired intervals and any predeclared harm trigger.",
            limitations="Four captures are from previous tick; current tick newly executes Y36 two full passes "
                        "and completes six pair comparisons. One training seed; paired intervals are not seed uncertainty.")
        output_ref = save(path, result)
        for name, cid in (("y52", "AdaRES-Y52"), ("y54", "AdaRES-Y54")):
            candidate = next(c for c in work["candidates"] if c["id"] == cid)
            assert candidate["stage"] == "validation"
            bind(candidate, ref(paths["y36"]), output_ref, ref(__file__))
            candidate.update(stage="complete", scientific_status="fixed30_and_full_frozen_comparison_finished",
                scientific_outcome=output_ref, failure_scope="implementation" if harms[name] else "inconclusive",
                next_action="Preserve same fixed30 checkpoints, FIRST evaluations/scene intervals/costs. "
                            "No automatic promotion, tuning, rerun or whole-mechanism falsification; "
                            + ("review predeclared subgroup harm." if harms[name] else "no subgroup harm trigger observed."))
        work.pop("design_cycle", None)
        lease()
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(review=output_ref, summary=result["summary"],
                              harm_counts={k: len(v) for k, v in harms.items()},
                              vs_y36={name: d["y36"]["all"] for name, d in differences.items()}), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
