"""Launch only the already prepared original Y96 fixed review from its canonically released validation stage."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,tick
import closure

def module(token):
    lease(token,300);w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y96')
    binding=tick.load(closure.artifact(ROOT,c['frozen_review_execution_binding']));assert binding['complete'] and binding['CPU_identity_and_launch_guards_passed']==11 and binding['source_files_verified']==26
    assert binding['original_candidate_commit']=='acb8c1769b7f3d66a92d287da8859a06a138b8b0' and binding['current_runtime_marker']=='6fb26984ae652efbba177ab20c39575c44a69517'
    source=(ROOT/'scripts/y99_frozen_review_job.py').read_text(encoding='utf-8')
    for old,new in [('AdaRES-Y99','AdaRES-Y96'),('adares_y99_spatial_mask_skip','adares_y96_seed_query_skip'),('y99_','y96_'),('53f5b028bdf24e8b41dd934931d1fedc99843d8d','acb8c1769b7f3d66a92d287da8859a06a138b8b0'),('2980836','3893460'),('20261111_180638','20261110_165702')]:
        assert old in source;source=source.replace(old,new)
    start=source.index('def launch(');end=source.index('def observe(',start);before=source[start:end];changed=before
    needle="c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot'";assert changed.count(needle)==1;changed=changed.replace(needle,"c['stage']=='validation' and c['required_server']=='server-1' and c['required_gpu']==0")
    needle="assert c['stage']=='blocked' and not c.get('frozen_review_job')";assert changed.count(needle)==1;changed=changed.replace(needle,"assert c['stage']=='validation' and not c.get('frozen_review_job')")
    source=source[:start]+changed+source[end:]
    namespace={'__name__':'y96_original_review','__file__':__file__};exec(compile(ast.parse(source),__file__,'exec'),namespace);return namespace

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('launch','observe'));parser.add_argument('--token',required=True);parser.add_argument('--label',default='start');a=parser.parse_args();n=module(a.token)
    if a.operation=='observe':n['observe'](a.token,a.label)
    else:n['launch'](a.token)
