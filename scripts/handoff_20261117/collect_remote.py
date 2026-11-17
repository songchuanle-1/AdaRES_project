"""Read-only source/environment inventory for the original AdaRES server."""

import argparse
import base64
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tarfile


ROOT = Path("/bak_home/songcl/Project/AdaRES_project/AdaRES")
PYTHON = "/home/songcl/.conda/envs/adares_project/bin/python"
SOURCE_DIRS = {"ipdn", "pointnet2", "segmentator", "sng_parser", "src", "tools",
               "configs_song"}
SOURCE_EXTENSIONS = {".py", ".sh", ".yaml", ".yml", ".cpp", ".cc", ".c", ".cu",
                     ".h", ".hpp", ".cuh", ".cmake", ".cfg", ".ini", ".toml",
                     ".md", ".rst"}
SKIP_DIRS = {"__pycache__", ".git", "build", "dist", ".pytest_cache"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def command(args):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=30, check=False)
    return {"returncode": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace")}


def sources():
    selected = []
    links = []
    native = []
    for base, dirs, names in os.walk(str(ROOT), followlinks=False):
        parent = Path(base)
        rel_parent = parent.relative_to(ROOT)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                         and not d.endswith(".egg-info")
                         and (parent != ROOT or d in SOURCE_DIRS))
        for name in list(dirs):
            path = parent / name
            if path.is_symlink():
                links.append({"path": path.relative_to(ROOT).as_posix(),
                              "target": os.readlink(str(path))})
                dirs.remove(name)
        for name in sorted(names):
            path = parent / name
            relative = path.relative_to(ROOT).as_posix()
            if path.is_symlink():
                links.append({"path": relative, "target": os.readlink(str(path))})
                continue
            if path.suffix == ".so":
                native.append({"path": relative, "bytes": path.stat().st_size,
                               "sha256": digest(path.read_bytes())})
            if path.suffix not in SOURCE_EXTENSIONS and not (
                name.startswith(("requirements", "LICENSE", "COPYING"))
                or name == "CMakeLists.txt"
            ):
                continue
            data = path.read_bytes()
            selected.append((relative, data, path.stat().st_mode & 0o777))
    for name in ("backbones",):
        path = ROOT / name
        if path.is_symlink():
            links.append({"path": name, "target": os.readlink(str(path))})
    return selected, links, native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != "dl01-MS-7C60" or sys.executable != PYTHON:
        raise SystemExit("Unexpected source server or interpreter")
    packages = sorted(
        [{"name": dist.metadata.get("Name", ""), "version": dist.version}
         for dist in importlib.metadata.distributions()],
        key=lambda row: (row["name"].lower(), row["version"]),
    )
    selected, links, native = sources()
    log = ROOT / "logs/train_adares_y102_geometry_mask_skip_res_20261112_235503.log"
    log_data = log.read_bytes()
    progress = re.findall(rb"(Train|Val)\[(\d+)/(\d+)\].*?(\d+)/(\d+)", log_data)
    proc = Path("/proc/4747")
    body = {
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "code_root": str(ROOT),
        "executable": sys.executable,
        "python_version": sys.version,
        "source_marker": (ROOT / ".codex_source_commit").read_text().strip(),
        "uname": command(["uname", "-a"]),
        "nvcc": command(["/usr/local/cuda/bin/nvcc", "--version"])
        if Path("/usr/local/cuda/bin/nvcc").exists() else None,
        "gpu0": command(["nvidia-smi", "-i", "0",
                         "--query-gpu=name,uuid,driver_version,power.limit",
                         "--format=csv,noheader"]),
        "packages": packages,
        "symlinks_not_dereferenced": links,
        "native_binaries_not_in_git": native,
        "sources": [{"path": name, "bytes": len(data), "sha256": digest(data),
                     "mode": oct(mode)} for name, data, mode in selected],
        "y102": {
            "pid": 4747, "pid_exists": proc.exists(),
            "cmdline": (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if proc.exists() else None,
            "process_state": (proc / "status").read_text().splitlines()[:8]
            if proc.exists() else None,
            "log": str(log), "log_bytes": len(log_data),
            "log_sha256": digest(log_data),
            "latest_native_counter": [part.decode() for part in progress[-1]]
            if progress else None,
            "progress_confirmed_this_capture": False,
            "no_signal_restart_checkpoint_load_or_gpu_execution": True,
        },
    }
    if args.sources:
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name, data, mode in selected:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = mode
                info.mtime = 0
                archive.addfile(info, io.BytesIO(data))
        payload = stream.getvalue()
        body["source_archive_sha256"] = digest(payload)
        body["source_archive_base64"] = base64.b64encode(payload).decode("ascii")
    print(json.dumps(body, ensure_ascii=False))


if __name__ == "__main__":
    main()
