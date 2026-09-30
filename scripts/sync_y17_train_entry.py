"""One reviewed train-entry update; refuse any unrecognized parent or bytes."""

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml


BASE = "2bb1c107b27fbfdbdf2d6ffd25b43bb5bf4f3e6c"
BEFORE = {
    "427152ba34fe936dbe31b6e7c0a1992f3a00d04f1a1de0353e8ad4d99ed5151e",
    "500b6f2ba20fbbc990609fdb6c47948e6b43c0eaf2c6e2d6b8086a754f146fa3",
}
AFTER = "04afc00889dc58e439197a04be50457664a7906d42d7badfdc96033f9d99b41e"
REMOTE = r'''
import base64, fcntl, hashlib, json, os, pathlib, sys, tempfile
d=json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
root=pathlib.Path(d["root"])
lock=open(root/".codex_train_entry_sync.lock","a")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (root/".codex_source_commit").read_text().strip()==d["base"], "marker changed"
path=root/"tools/train.py"
old=path.read_bytes()
digest=lambda b: hashlib.sha256(b).hexdigest()
assert digest(old) in d["before"] or digest(old)==d["after"], "unreviewed remote train"
new=base64.b64decode(d["data"])
assert digest(new)==d["after"], "payload mismatch"
if old != new:
    fd,tmp=tempfile.mkstemp(prefix=".y17_train_",dir=str(path.parent))
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(new); f.flush(); os.fsync(f.fileno())
        os.chmod(tmp,path.stat().st_mode)
        assert path.read_bytes()==old, "remote train changed during sync"
        os.replace(tmp,path)
        assert digest(path.read_bytes())==d["after"], "final hash mismatch"
    except BaseException:
        if path.read_bytes()==new:
            fd,restore=tempfile.mkstemp(prefix=".y17_restore_",dir=str(path.parent))
            with os.fdopen(fd,"wb") as f: f.write(old); f.flush(); os.fsync(f.fileno())
            os.chmod(restore,path.stat().st_mode)
            os.replace(restore,path)
        raise
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
print("Y17_TRAIN_ENTRY_HASH_VERIFIED "+digest(path.read_bytes()))
'''


def main():
    project, target, root, base, head = sys.argv[1:]
    if base != BASE:
        raise ValueError("Train entry update requires the reviewed Y15 parent")
    path = "AdaRES/tools/train.py"
    old = subprocess.check_output(["git", "-C", project, "show", base+":"+path])
    new = subprocess.check_output(["git", "-C", project, "show", head+":"+path])
    sha = lambda b: hashlib.sha256(b).hexdigest()
    assert sha(old) in BEFORE and sha(new) == AFTER
    payload = base64.b64encode(json.dumps(
        dict(root=root, base=base, before=sorted(BEFORE), after=AFTER,
             data=base64.b64encode(new).decode())).encode()).decode()
    servers = yaml.safe_load((Path(project)/"SERVERS.yaml").read_text(encoding="utf-8"))
    server = servers["servers"]["server-2"]
    assert target == server["user"]+"@"+server["host"]
    assert root == server["remote_code_root"]
    program = REMOTE.replace("sys.argv[1]", repr(payload))
    subprocess.run(["ssh", "-o", "BatchMode=yes", target,
                    server["python"]+" -"], input=program.encode(),
                   check=True, timeout=40)


if __name__ == "__main__":
    main()
