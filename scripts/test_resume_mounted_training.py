"""CPU-only identity regression tests; no remote access or training."""

from copy import deepcopy
import unittest

from resume_mounted_training import restored_task, SPECS


class RestoredTaskTests(unittest.TestCase):
    def fixture(self, candidate_id):
        stem, gpu, _ = SPECS[candidate_id]
        config = "configs_song/model_paper5/" + stem + ".yaml"
        experiment = dict(
            id=candidate_id, status="stopped_unknown", server="server-2",
            gpu=gpu, config=config, pid=1234, log="/preserved.log")
        proof = dict(pid=1234, log="/preserved.log", cmdline=[
            "python", "-S", "tools/train.py", config,
            "--num_gpus", "1", "--gpu_ids", gpu])
        receipt = dict(id=candidate_id, config=config, code_commit="a" * 40)
        return experiment, proof, receipt

    def test_both_frozen_placements_and_protocols(self):
        for candidate_id in SPECS:
            with self.subTest(candidate_id=candidate_id):
                result = restored_task(candidate_id, *self.fixture(candidate_id))
                self.assertEqual(result["gpu"], SPECS[candidate_id][1])
                self.assertEqual(result["fixed_epochs"], 30)
                self.assertEqual(result["min_epochs"], 30)
                self.assertEqual(result["total_epochs"], 30)
                self.assertEqual(result["status"], "stopped_unknown")

    def test_live_or_completed_runs_are_not_restarted(self):
        for status in ("running", "claiming", "done_complete"):
            fixture = self.fixture("AdaRES-Y52")
            fixture[0]["status"] = status
            with self.subTest(status=status), self.assertRaises(AssertionError):
                restored_task("AdaRES-Y52", *fixture)

    def test_changed_attempt_identity_is_rejected(self):
        for key, value in (("pid", 9999), ("log", "/different.log"), ("gpu", "0"),
                           ("server", "server-6"), ("config", "different.yaml")):
            fixture = deepcopy(self.fixture("AdaRES-Y52"))
            fixture[0][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                restored_task("AdaRES-Y52", *fixture)

    def test_multigpu_proof_is_rejected(self):
        fixture = self.fixture("AdaRES-Y54")
        fixture[1]["cmdline"][-3] = "2"
        with self.assertRaises(AssertionError):
            restored_task("AdaRES-Y54", *fixture)


if __name__ == "__main__":
    unittest.main()
