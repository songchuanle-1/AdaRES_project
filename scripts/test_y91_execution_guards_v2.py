"""Correct a generated negative fixture; retain original helpers unchanged."""
import unittest
import test_y91_execution_guards as original
import advance_y91 as advance
import y91_deployment as deploy
from prepare_y60_stage_heads import ROOT, tick


class ExecutionGuards(original.ExecutionGuards):
    def test_modified_or_unrelated_source_refused(self):
        rows = [["A", path] for path in deploy.paths("AdaRES-Y91")]
        for bad in (rows[:-1], rows + [["A", "other.py"]], [["M", rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, "AdaRES-Y91")
        with self.assertRaises(AssertionError):
            deploy.validate_delta(rows, "AdaRES-Y92")

    def test_both_native_payloads_remain_bound(self):
        super().test_both_native_payloads_remain_bound()
        candidate, _, _, frozen = advance.prepared(tick.load(ROOT / "pipeline/research_work.json"))
        self.assertEqual(candidate["code_commit"], deploy.Y91)
        self.assertEqual(frozen["runtime_server"], "server-2")
        self.assertEqual(frozen["runtime_physical_gpu"], 0)
        self.assertEqual(frozen["runtime_power_limit_watts"], 280.)


if __name__ == "__main__":
    unittest.main()
