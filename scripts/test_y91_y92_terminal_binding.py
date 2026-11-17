"""Actual source/native bindings and negative guards, without checkpoint loading."""
import copy
import unittest
from prepare_y60_stage_heads import ROOT, monitor, tick
import y91_y92_terminal_contract as binding
import y92_deployment as deploy


class AdditiveBinding(unittest.TestCase):
    def task_and_server(self):
        return (dict(binding.SPEC, status="running", launch_verified=True, gpu=0),
                monitor.load_servers(str(ROOT))["server-2"])

    def test_original_run_and_checkpoint_directory(self):
        self.assertEqual(binding.SPEC["pid"], 83827)
        self.assertEqual(binding.SPEC["code_commit"], deploy.Y91)
        self.assertEqual(binding.CHECKPOINT_DIRECTORY, "20261107_102124")
        self.assertEqual(binding.SPEC["admission_sha256"],
                         "4d59825df7e4d0cfe428fc47c3ff6102749f2b1e5f12d84b71aaf5238aa96eaf")
        self.assertEqual(binding.SPEC["freeze_sha256"],
                         "e5e6eef716ac892e827fe9131907ec44add8fdad1f619da8a568dacfc6935ca1")

    def test_only_explicit_original_and_tested_additive_markers(self):
        for marker in (deploy.Y91, deploy.Y92):
            binding.validate_marker(marker, "AdaRES-Y91")
        for marker in (deploy.Y90, "0" * 40):
            with self.assertRaises(AssertionError):
                binding.validate_marker(marker, "AdaRES-Y91")
        with self.assertRaises(AssertionError):
            binding.validate_marker(deploy.Y92, "AdaRES-Y92")

    def test_unchanged_original_source_and_exact_new_additions(self):
        deployments = deploy.contract(deploy.Y91, deploy.Y92, "AdaRES-Y91")
        self.assertEqual(list(deployments), [deploy.Y91, deploy.Y92])
        self.assertEqual(len(deployments[deploy.Y92]["source_files"]), 6)
        self.assertEqual(len(deployments[deploy.Y92]["configs"]), 1)
        rows = [row.split("\t") for row in deploy.git(
            "diff", "--no-renames", "--name-status", deploy.Y91, deploy.Y92).decode().splitlines()]
        deploy.validate_delta(rows, "AdaRES-Y92")

    def test_actual_native_bound_additive_contract(self):
        task, server = self.task_and_server()
        self.assertEqual(binding.build_contract(task, server)["allowed_markers"],
                         [deploy.Y91, deploy.Y92])

    def test_original_run_identity_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in (("id", "AdaRES-Y92"), ("pid", 1), ("gpu", 1),
                           ("server", "server-1"), ("code_commit", deploy.Y92),
                           ("config", "other.yaml"), ("log", "/wrong/run.log"),
                           ("status", "done_complete"), ("launch_verified", False)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                binding.build_contract(dict(task, **{key: value}), server)

    def test_original_runtime_and_data_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in (("python", "/wrong/python"), ("site_packages", "/wrong/site"),
                           ("pointnet_egg", "/wrong/egg"), ("allowed_gpus", [0, 1]),
                           ("data_root", "/wrong/data/"), ("work_dir", "/wrong/checkpoints"),
                           ("expected_hostname", "foreign"),
                           ("runtime_path_mappings", [{"source": "/old", "target": "/new"}])):
            with self.subTest(key=key), self.assertRaises((ValueError, AssertionError)):
                binding.build_contract(task, dict(server, **{key: value}))

    def test_actual_native_reference_required_without_fabricated_result(self):
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y92")
        self.assertEqual(binding.native_evidence(work), candidate["native_smoke_result"])
        changed = copy.deepcopy(work)
        next(row for row in changed["candidates"]
             if row["id"] == "AdaRES-Y92").pop("native_smoke_result")
        with self.assertRaisesRegex(AssertionError, "Actual additive Y92 native evidence"):
            binding.native_evidence(changed)

    def test_foreign_or_modified_additions_refused(self):
        rows = [["A", path] for path in deploy.paths("AdaRES-Y92")]
        for bad in (rows + [["A", "other.py"]], [["M", rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, "AdaRES-Y92")


if __name__ == "__main__":
    unittest.main()
