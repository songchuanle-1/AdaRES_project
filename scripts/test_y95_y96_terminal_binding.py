"""Actual native evidence and original Y95 lifecycle compatibility."""
import copy
import unittest
from prepare_y60_stage_heads import ROOT, monitor, tick
import y95_y96_terminal_contract as binding
import y96_deployment as deploy


class AdditiveBinding(unittest.TestCase):
    def task_and_server(self):
        return (dict(binding.SPEC, status="running", launch_verified=True, gpu=0),
                monitor.load_servers(str(ROOT))["server-2"])

    def test_original_identity_and_directory(self):
        self.assertEqual(binding.SPEC["pid"], 34484)
        self.assertEqual(binding.CHECKPOINT_DIRECTORY, "20261109_081912")
        self.assertEqual(binding.SPEC["admission_sha256"],
                         "de4e2620341bcd651a6d2ad3d8bfa98dc7e2e8523a7da9f1ffaea8c11d7597c5")

    def test_only_original_and_native_tested_markers(self):
        for marker in (deploy.Y95, deploy.Y96):
            binding.validate_marker(marker, "AdaRES-Y95")
        for marker, cid in ((deploy.Y94, "AdaRES-Y95"), ("0" * 40, "AdaRES-Y95"),
                            (deploy.Y96, "AdaRES-Y96")):
            with self.assertRaises(AssertionError):
                binding.validate_marker(marker, cid)

    def test_exact_seven_additions_and_preserved_original(self):
        result = deploy.contract(deploy.Y95, deploy.Y96, "AdaRES-Y95")
        self.assertEqual(list(result), [deploy.Y95, deploy.Y96])
        self.assertEqual(len(result[deploy.Y96]["source_files"]), 6)
        self.assertEqual(len(result[deploy.Y96]["configs"]), 1)

    def test_actual_native_bound_contract(self):
        task, server = self.task_and_server()
        self.assertEqual(binding.build_contract(task, server)["allowed_markers"],
                         [deploy.Y95, deploy.Y96])

    def test_owner_changes_refused(self):
        task, server = self.task_and_server()
        for key, value in (("id", "AdaRES-Y96"), ("pid", 1), ("gpu", 1),
                           ("server", "server-1"), ("code_commit", deploy.Y96),
                           ("config", "other.yaml"), ("log", "/wrong/run.log"),
                           ("status", "done_complete"), ("launch_verified", False)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                binding.build_contract(dict(task, **{key: value}), server)

    def test_runtime_changes_refused(self):
        task, server = self.task_and_server()
        for key, value in (("python", "/wrong/python"), ("allowed_gpus", [0, 1]),
                           ("work_dir", "/wrong/checkpoints"), ("expected_hostname", "foreign"),
                           ("remote_code_root", "/wrong/code"),
                           ("runtime_path_mappings", [{"source": "/a", "target": "/b"}])):
            with self.subTest(key=key), self.assertRaises((AssertionError, ValueError)):
                binding.build_contract(task, dict(server, **{key: value}))

    def test_actual_native_evidence_required(self):
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y96")
        self.assertEqual(binding.native_evidence(work), candidate["native_smoke_result"])
        changed = copy.deepcopy(work)
        next(row for row in changed["candidates"] if row["id"] == "AdaRES-Y96").pop("native_smoke_result")
        with self.assertRaisesRegex(AssertionError, "Actual additive Y96"):
            binding.native_evidence(changed)

    def test_missing_foreign_modified_paths_refused(self):
        rows = [["A", path] for path in deploy.paths("AdaRES-Y96")]
        deploy.validate_delta(rows, "AdaRES-Y96")
        for bad in (rows[:-1], rows + [["A", "other.py"]], [["M", rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, "AdaRES-Y96")
