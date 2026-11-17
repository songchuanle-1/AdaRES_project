"""CPU regression tests for exact original-attempt recovery boundaries."""

from copy import deepcopy
import unittest

from resume_y56_y57_after_reboot import SPECS, checkpoint_matches, validate_task


class ResumeIdentityTests(unittest.TestCase):
    def fixture(self, identity):
        spec = SPECS[identity]
        cfg = "configs_song/model_paper5/" + spec["package"] + "_res.yaml"
        row = dict(id=identity, server="server-2", gpu=spec["gpu"], status="running",
                   pid=spec["pid"], log="/code/logs/" + spec["log"], config=cfg,
                   code_commit=spec["commit"], fixed_epochs=30, min_epochs=30, patience=30)
        proof = dict(pid=spec["pid"], log=row["log"], process_live=True,
                     cmdline=["python", "tools/train.py", cfg, "--num_gpus", "1",
                              "--gpu_ids", spec["gpu"]])
        receipt = dict(id=identity, config=cfg, code_commit=spec["commit"])
        return row, deepcopy(row), receipt, proof

    def test_both_original_runs_are_accepted(self):
        for identity in SPECS:
            validate_task(identity, *self.fixture(identity))

    def test_changed_protocol_identity_or_completed_run_is_refused(self):
        for identity in SPECS:
            for key, value in (("status", "done_complete"), ("status", "claiming"),
                               ("fixed_epochs", 31), ("patience", 10), ("min_epochs", 0),
                               ("total_epochs", 31), ("gpu", "7"), ("pid", 42),
                               ("code_commit", "0" * 40), ("resume_attempt_token", "prior")):
                records = self.fixture(identity)
                records[0][key] = value
                with self.subTest(identity=identity, key=key), self.assertRaises(AssertionError):
                    validate_task(identity, *records)

    def test_foreign_proof_is_refused(self):
        records = self.fixture("AdaRES-Y56")
        records[-1]["cmdline"][-1] = "1"
        with self.assertRaises(AssertionError):
            validate_task("AdaRES-Y56", *records)

    def test_only_exact_original_last_epoch_is_accepted(self):
        server = dict(work_dir="/work")
        for identity, spec in SPECS.items():
            row = dict(checkpoint="/work/" + spec["package"] + "_res/" + spec["run"] + "/last.pth",
                       saved_epoch=spec["epoch"], resume_epoch=spec["epoch"] + 1,
                       optimizer_state_count=1, scheduler=dict(last_epoch=spec["epoch"]))
            checkpoint_matches(identity, row, server)
            for key, value in (("checkpoint", "/work/older/last.pth"), ("saved_epoch", 1),
                               ("resume_epoch", 1), ("optimizer_state_count", 0),
                               ("scheduler", dict(last_epoch=1))):
                changed = {**row, key: value}
                with self.subTest(identity=identity, key=key), self.assertRaises(AssertionError):
                    checkpoint_matches(identity, changed, server)


if __name__ == "__main__":
    unittest.main()
