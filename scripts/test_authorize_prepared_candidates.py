import contextlib
from datetime import datetime, timedelta, timezone
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import authorize_prepared_candidates as scoped


class ScopedCommitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        root_patch = patch.object(scoped, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        env_patch = patch.dict(os.environ, {"CODEX_THREAD_ID": scoped.THREAD})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        scoped.git("init", "-q")
        scoped.git("config", "user.name", "Scoped Commit Test")
        scoped.git("config", "user.email", "test@example.invalid")
        scoped.git("config", "commit.gpgsign", "false")
        (self.root / "base.txt").write_text("base\n", encoding="utf-8")
        scoped.git("add", "base.txt")
        scoped.git("commit", "-q", "-m", "Initial isolated fixture")
        self.headless = self.root / "pipeline/headless"
        self.headless.mkdir(parents=True)
        (self.headless / ".foreground.lockdir").mkdir()
        (self.root / "pipeline/analyses").mkdir()
        (self.root / "pipeline/authorizations").mkdir()
        for name in (".tick_context.json", ".tick_context_state.next.json"):
            scoped.tick.save(self.headless / name, {"mode": "steady"})
        scoped.tick.save(self.headless / ".foreground.lockdir/owner.json", {
            "token": "owned", "thread_id": scoped.THREAD,
            "deadline_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "context_sha256": scoped.tick.digest(self.headless / ".tick_context.json"),
            "pending_sha256": scoped.tick.digest(self.headless / ".tick_context_state.next.json"),
        })
        package, bundle_path = scoped.SPECS["AdaRES-Y52"]
        files = [f"AdaRES/ipdn/model_paper5/{package}/{name}" for name in (
            "__init__.py", "backbone.py", "dec.py", "ipdn.py", "loss.py", "sample_model.py")]
        files.append(f"AdaRES/configs_song/model_paper5/{package}_res.yaml")
        for name in files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("exact tested bytes: " + name + "\n").encode())
        self.rows = [scoped.ref(path) for path in files]
        scoped.tick.save(self.root / bundle_path, {
            "complete": True, "id": "AdaRES-Y52", "files": self.rows,
        })
        scoped.tick.save(self.root / "pipeline/research_work.json", {"candidates": [{
            "id": "AdaRES-Y52", "stage": "blocked", "blocker_kind": "human_authorization",
            "blocker": "Scoped permission needed", "retry_after": "2099-01-01T00:00:00+00:00",
            "unblock_condition": "Owning user's confirmation", "next_action": "Confirm exact files",
            "evidence": [scoped.ref(bundle_path)],
        }]})

    def run_operation(self, operation, token="owned"):
        with contextlib.redirect_stdout(io.StringIO()):
            return scoped.apply("AdaRES-Y52", token, operation)

    def test_commit_preserves_unrelated_staged_and_working_changes(self):
        unrelated = self.root / "base.txt"
        unrelated.write_text("user staged change\n", encoding="utf-8")
        scoped.git("add", "base.txt")
        unrelated.write_text("user working change\n", encoding="utf-8")
        before = scoped.unrelated_index([row["path"] for row in self.rows])
        self.run_operation("authorize")
        result = self.run_operation("commit")
        self.assertEqual(result["committed_files"], 7)
        self.assertEqual(scoped.unrelated_index([row["path"] for row in self.rows]), before)
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "user working change\n")
        self.assertEqual(scoped.git("diff", "--cached", "--name-only").decode().strip(), "base.txt")
        changes = scoped.git("diff-tree", "--no-commit-id", "--name-only", "-r", result["code_commit"])
        self.assertEqual(set(changes.decode().splitlines()), {row["path"] for row in self.rows})

    def test_changed_tested_bytes_fail_before_authorization(self):
        (self.root / self.rows[0]["path"]).write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tested bundle"):
            self.run_operation("authorize")
        self.assertFalse((self.root / "pipeline/authorizations/AdaRES-Y52_commit.json").exists())
        self.assertEqual(scoped.git("diff", "--cached", "--name-only"), b"")

    def test_wrong_lease_and_replay_cannot_commit(self):
        with self.assertRaisesRegex(ValueError, "foreground lease"):
            self.run_operation("authorize", "not-owned")
        self.run_operation("authorize")
        self.run_operation("commit")
        head = scoped.git("rev-parse", "HEAD")
        with self.assertRaisesRegex(ValueError, "starting commit changed"):
            self.run_operation("commit")
        self.assertEqual(scoped.git("rev-parse", "HEAD"), head)


if __name__ == "__main__":
    unittest.main()
