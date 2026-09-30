"""Exercise the exact remote atomic-update body on an isolated temporary root."""

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import argparse

from sync_y17_train_entry import REMOTE


def main():
    with tempfile.TemporaryDirectory(prefix="y17_sync_test_") as temp:
        root = Path(temp)
        (root/"tools").mkdir()
        (root/".codex_source_commit").write_text("parent")
        path = root/"tools/train.py"
        old, new = b"old reviewed entry", b"new reviewed entry"
        digest = lambda b: hashlib.sha256(b).hexdigest()
        data = dict(root=temp, base="parent", before=[digest(old)],
                    after=digest(new), data=base64.b64encode(new).decode())
        def run():
            encoded = base64.b64encode(json.dumps(data).encode()).decode()
            return subprocess.run([sys.executable, "-", encoded],
                                  input=REMOTE.encode(), capture_output=True)
        path.write_bytes(b"foreign change")
        assert run().returncode != 0 and path.read_bytes() == b"foreign change"
        path.write_bytes(old)
        assert run().returncode == 0 and path.read_bytes() == new
        assert run().returncode == 0 and path.read_bytes() == new
        data["base"] = "wrong parent"
        assert run().returncode != 0 and path.read_bytes() == new
        assert (root/".codex_source_commit").read_text() == "parent"
        result = dict(passed=True, python=sys.version, cases=[
            "foreign_bytes_rejected", "atomic_update", "idempotent",
            "stale_marker_rejected", "marker_not_advanced"])
        parser = argparse.ArgumentParser()
        parser.add_argument("--output")
        args = parser.parse_args()
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
