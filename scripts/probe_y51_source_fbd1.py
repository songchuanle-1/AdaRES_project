"""Read exact native W2 confidence and loss ASTs, without importing the model."""
import ast,hashlib,json
from pathlib import Path
assert P['physical_gpu'] is None
root=Path(P['root'])
package=root/'ipdn/model_paper5/adares_w2_pointmass_competition'
files={}
methods=[]
for name in ('ipdn.py','loss.py','dec.py'):
    path=package/name
    text=path.read_text()
    files[name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    lines=text.splitlines()
    for cls in ast.parse(text).body:
        if not isinstance(cls,ast.ClassDef):continue
        for node in cls.body:
            if not isinstance(node,ast.FunctionDef):continue
            source='\n'.join(lines[node.lineno-1:node.end_lineno])
            if (name=='ipdn.py' and ('pred_indis' in source or 'indi' in node.name)) or (name=='loss.py' and 'indi' in node.name):
                methods.append(dict(file=name,cls=cls.name,name=node.name,source=source))
result=dict(complete=True,id='AdaRES-Y51',files=files,methods=methods,read_only=True,
            scientific_status='source_capture_only_not_gauge_test',no_model_change=True,no_GPU=True)
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
