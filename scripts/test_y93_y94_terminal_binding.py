"""Test actual Y94 evidence without loading or stopping the original Y93 run."""
import copy
import unittest

from prepare_y60_stage_heads import ROOT, monitor, tick
import y93_y94_terminal_contract as binding
import y94_deployment as deploy


class AdditiveBinding(unittest.TestCase):
    def task_and_server(self):
        return (dict(binding.SPEC, status="running", launch_verified=True, gpu=0),
                monitor.load_servers(str(ROOT))["server-2"])

    def test_original_run_and_directory(self):
        self.assertEqual(binding.SPEC["pid"], 47380)
        self.assertEqual(binding.SPEC["code_commit"], deploy.Y93)
        self.assertEqual(binding.CHECKPOINT_DIRECTORY, "20261108_082521")
        self.assertEqual(binding.SPEC["admission_sha256"],
                         "f6b87f2813f1e8e429772bc0f63963fe1a43e0582c6c65e21771092dcbf69832")
        self.assertEqual(binding.SPEC["freeze_sha256"],
                         "8563a67a55b91f7c72b2899d03917b28d322e2f5ac14130c016511c72b0c7ef8")

    def test_only_original_and_actual_native_additive_markers(self):
        for marker in (deploy.Y93, deploy.Y94):
            binding.validate_marker(marker, "AdaRES-Y93")
        for marker in (deploy.Y92, "0" * 40):
            with self.assertRaises(AssertionError):
                binding.validate_marker(marker, "AdaRES-Y93")
        with self.assertRaises(AssertionError):
            binding.validate_marker(deploy.Y94, "AdaRES-Y94")

    def test_exact_additions_and_original_sources(self):
        deployments = deploy.contract(deploy.Y93, deploy.Y94, "AdaRES-Y93")
        self.assertEqual(list(deployments), [deploy.Y93, deploy.Y94])
        self.assertEqual(len(deployments[deploy.Y94]["source_files"]), 6)
        self.assertEqual(len(deployments[deploy.Y94]["configs"]), 1)
        rows = [row.split("\t") for row in deploy.git(
            "diff", "--no-renames", "--name-status", deploy.Y93, deploy.Y94).decode().splitlines()]
        deploy.validate_delta(rows, "AdaRES-Y94")

    def test_actual_native_bound_contract(self):
        task, server = self.task_and_server()
        self.assertEqual(binding.build_contract(task, server)["allowed_markers"],
                         [deploy.Y93, deploy.Y94])

    def test_original_run_identity_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in (("id", "AdaRES-Y94"), ("pid", 1), ("gpu", 1),
                           ("server", "server-1"), ("code_commit", deploy.Y94),
                           ("config", "other.yaml"), ("log", "/wrong/run.log"),
                           ("status", "done_complete"), ("launch_verified", False)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                binding.build_contract(dict(task, **{key: value}), server)

    def test_private_runtime_and_exact_mapping_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in (("python", "/wrong/python"), ("allowed_gpus", [0, 1]),
                           ("work_dir", "/wrong/checkpoints"), ("expected_hostname", "foreign"),
                           ("remote_code_root", "/wrong/code"),
                           ("runtime_path_mappings", [{"source": "/old", "target": "/new"}])):
            with self.subTest(key=key), self.assertRaises((AssertionError, ValueError)):
                binding.build_contract(task, dict(server, **{key: value}))

    def test_actual_native_reference_required(self):
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y94")
        self.assertEqual(binding.native_evidence(work), candidate["native_smoke_result"])
        changed = copy.deepcopy(work)
        next(row for row in changed["candidates"]
             if row["id"] == "AdaRES-Y94").pop("native_smoke_result")
        with self.assertRaisesRegex(AssertionError, "Actual additive Y94 native evidence"):
            binding.native_evidence(changed)

    def test_foreign_or_modified_additions_refused(self):
        rows = [["A", path] for path in deploy.paths("AdaRES-Y94")]
        for bad in (rows + [["A", "other.py"]], [["M", rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, "AdaRES-Y94")


if __name__ == "__main__":
    unittest.main()
