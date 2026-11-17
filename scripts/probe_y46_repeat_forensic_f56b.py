"""Reproduce the failed first-scene cached-repeat check, without declaring PASS."""
import ast
import hashlib
import json
from pathlib import Path
source=Path('/tmp/adares_y46_controls_reference_f56b.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='d6914d9f758013dbff4f46796d0e02acc07585478f1f54cd2b76cd32125b252f'
tree=ast.parse(source.read_text())
body=[]
class Forensic(ast.NodeTransformer):
    def visit_Assert(self,node):
        if isinstance(node.test,ast.Call) and isinstance(node.test.func,ast.Attribute) and node.test.func.attr=='equal':
            return ast.parse("""
repeats=[head(query,values) for _ in range(8)]
result.setdefault('cached_repeat_forensics',[]).append(dict(exact_repeats=[bool(torch.equal(correct,v)) for v in repeats],max_errors=[float((correct-v).abs().max()) for v in repeats],max_absolute_score=float(correct.abs().max()),dtype=str(correct.dtype),component_rows=len(values[0]),readout='unchanged production index_add aggregation'))
""").body
        return node
for node in tree.body:
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='handle' and node.value.func.attr=='remove':break
    if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='i':
        node.iter=ast.List(elts=[ast.Constant(value=0)],ctx=ast.Load())
        node=Forensic().visit(node)
    body.append(node)
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(source),'exec'),globals())
result['complete']=True
result['forensic_only_not_repeat_pass']=True
result['native_repeat_gate']='FAILED_UNDER_ORIGINAL_EXACT_RULE'
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
