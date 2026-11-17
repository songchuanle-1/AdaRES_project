"""Repair payload field names; preserve the first failed read-only audit."""

import ast
from pathlib import Path

source_path = Path(__file__).with_name("capture_y36_launch.py")
source = source_path.read_bytes()


class Identity(ast.NodeTransformer):
    def visit_keyword(self, node):
        self.generic_visit(node)
        node.arg = {"y35_files": "y36_files", "y36_files": "y37_files"}.get(node.arg, node.arg)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            if node.value == "1":
                return ast.copy_location(ast.Constant(value="0"), node)
            value = node.value.replace("Y36", "Y37").replace("y36_", "y37_")
            value = value.replace("Y35", "Y36").replace("y35_", "y36_")
            value = value.replace("adares_y37_local_adjacency", "adares_y37_pair_distribution")
            value = value.replace("adares_y36_language_evidence", "adares_y36_local_adjacency")
            value = value.replace("108331", "100018")
            value = value.replace("argv[argv.index(b'--gpu_ids')+1]==b'1'",
                                  "argv[argv.index(b'--gpu_ids')+1]==b'0'")
            value = value.replace("old_argv[old_argv.index(b'--gpu_ids')+1]==b'0'",
                                  "old_argv[old_argv.index(b'--gpu_ids')+1]==b'1'")
            value = value.replace("'nvidia-smi','-i','1'", "'nvidia-smi','-i','0'")
            value = value.replace("actual_gpu=1", "actual_gpu=0")
            value = value.replace("gpu1_device", "gpu0_device")
            value = value.replace("preserved_y36_gpu=0", "preserved_y36_gpu=1")
            value = value.replace("Y36 source/PID/config/GPU0", "Y36 source/PID/config/GPU1")
            value = value.replace("pipeline/analyses/y37_launch_", "pipeline/analyses/y37_launch_v2_")
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"),
     {"__name__": "__main__", "__file__": str(source_path)})
