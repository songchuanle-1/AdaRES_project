"""Bind the unchanged frozen cost protocol to archived Y99 on its original GPU0."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,tick
import closure

def module(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(x for x in w['candidates'] if x['id']=='AdaRES-Y99');shared=next(x for x in w['candidates'] if x['id']=='AdaRES-Y101')
    assert candidate['code_commit']=='53f5b028bdf24e8b41dd934931d1fedc99843d8d' and '+8192/total164637829/' in candidate['matched_control']
    assert shared['stage']=='complete' and tick.load(closure.artifact(ROOT,shared['complete_scientific_review']))['full_cost_and_postaudit_complete']
    for key in ('frozen_review_reference_stage','frozen_review_pointnet_origin','frozen_review_runtime_binding'):
        closure.artifact(ROOT,shared[key])
    base=ROOT/'scripts/y101_frozen_trained_cost.py';source=base.read_text(encoding='utf-8')
    for old,new in [('AdaRES-Y101','AdaRES-Y99'),('adares_y101_query_ffn_skip/','adares_y99_spatial_mask_skip/'),('y101_','y99_'),('2263581','2980836'),('164695173','164637829')]:
        assert old in source;source=source.replace(old,new)
    for key in ('frozen_review_reference_stage','frozen_review_pointnet_origin','frozen_review_runtime_binding'):
        needle="c['"+key+"']";assert source.count(needle)==1;source=source.replace(needle,"shared['"+key+"']")
    needle="(root/'.codex_source_commit').read_text().strip()==P['code_commit']";assert source.count(needle)==1
    source=source.replace(needle,"(root/'.codex_source_commit').read_text().strip()==P['runtime_marker'];assert P['code_commit']=='53f5b028bdf24e8b41dd934931d1fedc99843d8d' and P['runtime_marker']=='6fb26984ae652efbba177ab20c39575c44a69517'")
    needle="code_commit=p['code_commit'],gpu=0";assert source.count(needle)==1;source=source.replace(needle,"code_commit=p['code_commit'],runtime_marker=p['runtime_marker'],gpu=0")
    needle="evidence=[ref(__file__),ref(source),ref(path)";assert source.count(needle)==1;source=source.replace(needle,"evidence=[ref(__file__),ref(base),ref(source),ref(path)")
    namespace={'__name__':'y99_cost','__file__':__file__,'shared':shared,'base':base}
    exec(compile(ast.parse(source),__file__,'exec'),namespace)
    return namespace

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('build','launch','observe'));parser.add_argument('--token',required=True);parser.add_argument('--label',default='start');a=parser.parse_args();m=module(a.token)
    if a.operation=='observe':m['observe'](a.token,a.label)
    else:m[a.operation](a.token)
