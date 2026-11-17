"""Run the immutable Y23 resume guard with compute queries scoped to GPU1."""

import ast
import hashlib
from pathlib import Path

path = Path(__file__).with_name("resume_y23_after_y18.py")
source = path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "811695561e8ae9dd1b75fec6244ccde70cc9436f63414a147ae0bcf827aba228"
tree = ast.parse(source, filename=str(path))
programs = [node for node in ast.walk(tree) if isinstance(node, ast.Constant)
            and isinstance(node.value, str) and "gpu_uuid=subprocess.check_output" in node.value]
assert len(programs) == 1
old = '["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"]'
new = '["nvidia-smi","-i","1","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"]'
assert programs[0].value.count(old) == 1
programs[0].value = programs[0].value.replace(old, new)
exec(compile(tree, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
