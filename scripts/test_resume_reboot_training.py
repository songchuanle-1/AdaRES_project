"""CPU-only regression checks for repeated-resume attempt binding."""

from copy import deepcopy
import unittest

from resume_reboot_training import restored_task, SPECS


class RebootResumeTests(unittest.TestCase):
    def fixture(self, candidate_id="AdaRES-Y52"):
        stem, gpu, _ = SPECS[candidate_id]
        config = "configs_song/model_paper5/" + stem + ".yaml"
        experiment = dict(
            id=candidate_id, status="running", server="server-2", gpu=gpu,
            config=config, code_commit="a" * 40, pid=1234, log="/stdout.log",
            resume="/old/last.pth", resume_attempt_token="prior-token", resume_epoch=24,
            resumed_work_dir="/latest/native", fixed_epochs=30,
            min_epochs=30, total_epochs=30, patience=30)
        launch = dict(
            token="prior-token", result=dict(pid=1234, log="/stdout.log"),
            payload=dict(config=config, gpu=gpu, checkpoint="/old/last.pth"))
        verification = dict(
            token="prior-token", result="resumed_training_progress_verified",
            pid=1234, gpu=gpu, resume_epoch=24, samples=[
                dict(pid=1234, restored=True, errors=[], progress=[24, step],
                     native_logs=["/latest/native/native.log"])
                for step in (127, 138)])
        receipt = dict(id=candidate_id, config=config, code_commit="a" * 40)
        return experiment, launch, verification, receipt

    def test_both_latest_verified_attempts_keep_protocol(self):
        for candidate_id in SPECS:
            with self.subTest(candidate_id=candidate_id):
                result = restored_task(candidate_id, *self.fixture(candidate_id))
                self.assertEqual(result["gpu"], SPECS[candidate_id][1])
                self.assertEqual(result["fixed_epochs"], 30)
                self.assertEqual(result["total_epochs"], 30)

    def test_completed_claiming_or_unfrozen_runs_are_rejected(self):
        for key, value in (("status", "done_complete"), ("status", "claiming"),
                           ("fixed_epochs", 31), ("min_epochs", 20),
                           ("total_epochs", 31), ("patience", 10)):
            fixture = self.fixture()
            fixture[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                restored_task("AdaRES-Y52", *fixture)

    def test_old_or_mismatched_attempt_is_rejected(self):
        for key, value in (("pid", 9999), ("log", "/older.log"),
                           ("resume_attempt_token", "older-token"),
                           ("resumed_work_dir", "/older/native"),
                           ("code_commit", "b" * 40), ("gpu", "0")):
            fixture = self.fixture()
            fixture[0][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                restored_task("AdaRES-Y52", *fixture)

    def test_unverified_or_nonprogressing_attempt_is_rejected(self):
        for mutation in ("result", "restored", "progress", "errors", "pid"):
            fixture = deepcopy(self.fixture())
            verification = fixture[2]
            if mutation == "result":
                verification["result"] = "pending_progress"
            elif mutation == "progress":
                verification["samples"][-1]["progress"] = [24, 127]
            elif mutation == "errors":
                verification["samples"][-1]["errors"] = ["Traceback"]
            elif mutation == "pid":
                verification["samples"][-1]["pid"] = 9999
            else:
                verification["samples"][-1]["restored"] = False
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                restored_task("AdaRES-Y52", *fixture)

    def test_archived_interruption_can_be_reconstructed(self):
        fixture = self.fixture()
        fixture[0]["status"] = "stopped_unknown"
        self.assertEqual(restored_task("AdaRES-Y52", *fixture)["status"], "stopped_unknown")


if __name__ == "__main__":
    unittest.main()
