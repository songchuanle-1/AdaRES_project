"""Source-only binding tests; no original Y90 checkpoint is loaded."""
import copy
import unittest
from prepare_y60_stage_heads import ROOT, monitor
import y90_y91_terminal_contract as binding
import y91_deployment as deploy


class AdditiveBinding(unittest.TestCase):
    def task_and_server(self):
        return (dict(binding.SPEC, status="running", launch_verified=True, gpu=0),
                monitor.load_servers(str(ROOT))["server-1"])

    def test_original_run_and_checkpoint_directory(self):
        self.assertEqual(binding.SPEC["pid"], 2442962)
        self.assertEqual(binding.SPEC["code_commit"], deploy.Y90)
        self.assertEqual(binding.CHECKPOINT_DIRECTORY, "20261107_071428")
        self.assertEqual(binding.SPEC["admission_sha256"],
                         "f790bc29637612751ae4e483a2ad21efa6c468d273ddadf4523ad182bb585f56")
        self.assertEqual(binding.SPEC["freeze_sha256"],
                         "ad60ab5a94db03b12405a043be738aae978786d560d3b707357150994ec95395")

    def test_only_explicit_original_and_tested_additive_markers(self):
        for marker in (deploy.Y90, deploy.Y91):
            binding.validate_marker(marker, "AdaRES-Y90")
        for marker in (deploy.Y89, "0" * 40):
            with self.assertRaises(AssertionError):
                binding.validate_marker(marker, "AdaRES-Y90")
        with self.assertRaises(AssertionError):
            binding.validate_marker(deploy.Y91, "AdaRES-Y91")

    def test_unchanged_original_source_and_exact_new_additions(self):
        deployments = deploy.contract(deploy.Y90, deploy.Y91, "AdaRES-Y90")
        self.assertEqual(set(deployments), {deploy.Y90, deploy.Y91})
        self.assertEqual(len(deployments[deploy.Y91]["source_files"]), 6)
        self.assertEqual(len(deployments[deploy.Y91]["configs"]), 1)

    def test_actual_native_bound_additive_contract(self):
        task, server = self.task_and_server()
        value = binding.build_contract(task, server)
        self.assertEqual(set(value["allowed_markers"]), {deploy.Y90, deploy.Y91})

    def test_original_run_identity_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in (("pid", 1), ("gpu", 1), ("server", "server-2"),
                           ("code_commit", deploy.Y91), ("log", "/wrong/run.log")):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                binding.build_contract(dict(task, **{key: value}), server)

    def test_original_runtime_and_mappings_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in (("python", "/wrong/python"), ("runtime_library_path", "/wrong/lib"),
                           ("allowed_gpus", [0, 1]), ("runtime_path_mappings", [])):
            changed = copy.deepcopy(server)
            changed[key] = value
            expected = ValueError if key == "allowed_gpus" else AssertionError
            with self.subTest(key=key), self.assertRaises(expected):
                binding.build_contract(task, changed)


if __name__ == "__main__":
    unittest.main()
