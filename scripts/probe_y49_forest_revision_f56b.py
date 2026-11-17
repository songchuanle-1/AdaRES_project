"""Exact forest elimination, with unchanged sparse-LU fallback for cyclic graphs."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import time
import zlib
import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
import torch
from torch import nn

reference_path=Path('/tmp/adares_y49_reference_f56b.py')
assert hashlib.sha256(reference_path.read_bytes()).hexdigest()=='b24ed8dbeed377e7f2c22bd4bd9647105254411275d9ce68d23d6a496692d2a0'
names=('inverse_diagonal','descriptors','GlobalResolventReadout')
nodes=[n for n in ast.parse(reference_path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
assert len(nodes)==3
namespace=dict(globals())
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(reference_path),'exec'),namespace)
reference_inverse=namespace['inverse_diagonal']
forest_counts=dict(forest_calls=0,cyclic_fallback_calls=0)

def forest_inverse_diagonal(graph,block_size=64):
    graph=graph.tocsr()
    n=graph.shape[0]
    assert (graph!=graph.T).nnz==0 and not graph.diagonal().any()
    components,_=connected_components(graph,directed=False)
    if graph.nnz//2 != n-components:
        forest_counts['cyclic_fallback_calls']+=1
        return reference_inverse(graph,block_size)
    assert np.all(graph.data==1)
    forest_counts['forest_calls']+=1
    parent=np.full(n,-2,dtype=np.int64)
    order=[]
    for root in range(n):
        if parent[root]!=-2:continue
        parent[root]=-1
        stack=[root]
        while stack:
            vertex=stack.pop()
            order.append(vertex)
            for neighbor in graph.indices[graph.indptr[vertex]:graph.indptr[vertex+1]]:
                if neighbor==parent[vertex]:continue
                assert parent[neighbor]==-2
                parent[neighbor]=vertex
                stack.append(int(neighbor))
    down=1.0+np.asarray(graph.sum(1)).ravel()
    for vertex in reversed(order):
        assert down[vertex]>0
        if parent[vertex]>=0:down[parent[vertex]]-=1.0/down[vertex]
    incoming=np.zeros(n)
    diagonal=np.empty(n)
    for vertex in order:
        denominator=down[vertex]-incoming[vertex]
        assert denominator>0
        diagonal[vertex]=1.0/denominator
        for neighbor in graph.indices[graph.indptr[vertex]:graph.indptr[vertex+1]]:
            if parent[neighbor]==vertex:
                incoming[neighbor]=1.0/(denominator+1.0/down[neighbor])
    assert np.isfinite(diagonal).all() and (diagonal>0).all()
    return diagonal

def unpack(item):
    raw=zlib.decompress(base64.b64decode(item['zlib_base64']))
    assert hashlib.sha256(raw).hexdigest()==item['sha256']
    return np.frombuffer(raw,dtype=np.dtype(item['dtype'])).reshape(item['shape']).copy()

assert P['physical_gpu'] is None
torch.set_num_threads(4)
capture_path=Path('/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json')
assert hashlib.sha256(capture_path.read_bytes()).hexdigest()=='5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'
capture=json.loads(capture_path.read_text())
torch.manual_seed(1999)
head=namespace['GlobalResolventReadout'](256).double()
rows=[]
for scene in capture['rows']:
    for expression in scene['expressions']:
        arrays={key:unpack(value) for key,value in expression['packed_native_inputs'].items()}
        masses=arrays['masses'].astype(np.float64)
        masks=arrays['masks']>0
        edges=arrays['graph_edges']
        graph=csr_matrix((np.ones(len(edges),bool),(edges[:,0],edges[:,1])),shape=(len(masses),len(masses)))
        namespace['inverse_diagonal']=reference_inverse
        start=time.monotonic()
        reference,partitions,unique=namespace['descriptors'](graph,masks,masses)
        original_seconds=time.monotonic()-start
        namespace['inverse_diagonal']=forest_inverse_diagonal
        start=time.monotonic()
        revised,new_partitions,new_unique=namespace['descriptors'](graph,masks,masses)
        revised_seconds=time.monotonic()-start
        bound=8*torch.finfo(torch.double).eps*max(1,len(masses))*max(1.0,float(reference.abs().max()),float(revised.abs().max()))
        error=float((reference-revised).abs().max())
        assert error<=bound and partitions==new_partitions and unique==new_unique
        query=torch.from_numpy(arrays['query'].astype(np.float64))
        head.zero_grad(set_to_none=True)
        scores=head(query,revised)
        logits=torch.stack((torch.zeros_like(scores),scores),-1)
        nn.functional.cross_entropy(logits,torch.arange(128).remainder(2)).backward()
        gradients={n:float(v.grad.norm()) for n,v in head.named_parameters()}
        assert len(gradients)==4 and all(np.isfinite(v) and v>0 for v in gradients.values())
        with torch.no_grad():
            assert torch.equal(scores,head(query,revised))
            wrong_gap=float((scores-head(query,revised.roll(1,dims=0))).abs().max())
        permutation_error=None
        if len(rows)<4:
            order=np.arange(len(masses)-1,-1,-1)
            permuted,_,_=namespace['descriptors'](graph[order][:,order],masks[:,order],masses[order])
            permutation_error=float((revised-permuted).abs().max())
            assert permutation_error<=bound
        rows.append(dict(dataset_index=scene['dataset_index'],ann_id=expression['ann_id'],nodes=len(masses),original_seconds=original_seconds,revised_seconds=revised_seconds,max_descriptor_error=error,existing8eps_nodes_bound=bound,partitions_exact=True,gradients=gradients,wrong_gap=wrong_gap,permutation_error=permutation_error))
        print('Y49_REV1_CPU',len(rows),round(original_seconds,3),round(revised_seconds,3),flush=True)
assert len(rows)==32
fallback=[]
for n in (4,8):
    graph=np.zeros((n,n))
    for i in range(n):graph[i,(i+1)%n]=graph[(i+1)%n,i]=1
    graph=csr_matrix(graph)
    actual=forest_inverse_diagonal(graph)
    expected=reference_inverse(graph)
    assert np.array_equal(actual,expected)
    fallback.append(dict(nodes=n,cyclic_fallback_exact=True))
result=dict(complete=True,id='AdaRES-Y49',implementation_revision=1,rows=rows,native_captured_expressions=32,forest_counts=forest_counts,cyclic_controls=fallback,total_original_seconds=sum(r['original_seconds'] for r in rows),total_revised_seconds=sum(r['revised_seconds'] for r in rows),maximum_descriptor_error=max(r['max_descriptor_error'] for r in rows),minimum_proxy_gradient=min(v for r in rows for v in r['gradients'].values()),limitations=['Exact mathematical readout implementation only; float64 comparison uses the already frozen8eps*nodes envelope.','No changed graph/masks/point masses/targets/head/model or validation selection.','All historical negative evidence and zero native equal-component-partition witnesses retained.','CPU proxy gradients are not revised native backward or production/resource admission.'])
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
