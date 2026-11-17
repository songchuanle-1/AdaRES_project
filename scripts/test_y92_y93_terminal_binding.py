"""Real native/source evidence and negative guards without loading Y92 checkpoints."""
import copy
import unittest

from prepare_y60_stage_heads import ROOT, monitor, tick
import y92_y93_terminal_contract as binding
import y93_deployment as deploy


class AdditiveBinding(unittest.TestCase):
    def task_and_server(self):
        return (dict(binding.SPEC, status="running", launch_verified=True, gpu=0),
                monitor.load_servers(str(ROOT))["server-1"])

    def test_original_run_and_directory(self):
        self.assertEqual(binding.SPEC["pid"], 1305468)
        self.assertEqual(binding.SPEC["code_commit"], deploy.Y92)
        self.assertEqual(binding.CHECKPOINT_DIRECTORY, "20261108_090202")
        self.assertEqual(binding.SPEC["admission_sha256"],
                         "56c0a076cc22dd82200f934781dd50b7a5972ab2adb3c383acb42cb6cb0822dc")
        self.assertEqual(binding.SPEC["freeze_sha256"],
                         "b866bf8a784b9b44ff63cb1dd7de74fd6288b0b84a1f2280a45c9eaa85a76f95")

    def test_only_original_and_actual_native_additive_markers(self):
        for marker in (deploy.Y92, deploy.Y93):
            binding.validate_marker(marker, "AdaRES-Y92")
        for marker in (deploy.Y91, "0" * 40):
            with self.assertRaises(AssertionError):
                binding.validate_marker(marker, "AdaRES-Y92")
        with self.assertRaises(AssertionError):
            binding.validate_marker(deploy.Y93, "AdaRES-Y93")

    def test_exact_additions_and_original_sources(self):
        deployments = deploy.contract(deploy.Y92, deploy.Y93, "AdaRES-Y92")
        self.assertEqual(list(deployments), [deploy.Y92, deploy.Y93])
        self.assertEqual(len(deployments[deploy.Y93]["source_files"]), 6)
        self.assertEqual(len(deployments[deploy.Y93]["configs"]), 1)
        rows = [row.split("\t") for row in deploy.git(
            "diff", "--no-renames", "--name-status", deploy.Y92, deploy.Y93).decode().splitlines()]
        deploy.validate_delta(rows, "AdaRES-Y93")

    def test_actual_native_bound_contract(self):
        task, server = self.task_and_server()
        self.assertEqual(binding.build_contract(task, server)["allowed_markers"],
                         [deploy.Y92, deploy.Y93])

    def test_original_run_identity_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in (("id", "AdaRES-Y93"), ("pid", 1), ("gpu", 1),
                           ("server", "server-2"), ("code_commit", deploy.Y93),
                           ("config", "other.yaml"), ("log", "/wrong/run.log"),
                           ("status", "done_complete"), ("launch_verified", False)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                binding.build_contract(dict(task, **{key: value}), server)

    def test_private_runtime_and_exact_mappings_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in (("python", "/wrong/python"),
                           ("runtime_library_path", "/wrong/cuda"),
                           ("allowed_gpus", [0, 1]), ("work_dir", "/wrong/checkpoints"),
                           ("expected_hostname", "foreign"), ("runtime_path_mappings", [])):
            with self.subTest(key=key), self.assertRaises((AssertionError, ValueError)):
                binding.build_contract(task, dict(server, **{key: value}))

    def test_actual_native_reference_required(self):
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y93")
        self.assertEqual(binding.native_evidence(work), candidate["native_smoke_result"])
        changed = copy.deepcopy(work)
        next(row for row in changed["candidates"]
             if row["id"] == "AdaRES-Y93").pop("native_smoke_result")
        with self.assertRaisesRegex(AssertionError, "Actual additive Y93 native evidence"):
            binding.native_evidence(changed)

    def test_foreign_or_modified_additions_refused(self):
        rows = [["A", path] for path in deploy.paths("AdaRES-Y93")]
        for bad in (rows + [["A", "other.py"]], [["M", rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, "AdaRES-Y93")


if __name__ == "__main__":
    unittest.main()
