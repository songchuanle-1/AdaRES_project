"""Advance archived Y94 on its exact original resource using preserved fixed review science."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,tick
import closure

def module(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y88')
    assert c['code_commit']=='cb4a5813755d38b3d1593bb803726aabd7d0a79b' and c['required_server']=='server-1' and c['required_gpu']==0
    t=tick.load(closure.artifact(ROOT,c['terminal_evidence']));assert t['checkpoints'][1]['epoch']==21 and '/20261106_053553/' in t['checkpoints'][1]['path'];closure.artifact(ROOT,c['terminal_archive_binding'])
    source=(ROOT/'scripts/y99_frozen_review_job.py').read_text(encoding='utf-8')
    for old,new in [('AdaRES-Y99','AdaRES-Y88'),('adares_y99_spatial_mask_skip','adares_y88_mask_query_quadratic'),('y99_','y88_'),('53f5b028bdf24e8b41dd934931d1fedc99843d8d','cb4a5813755d38b3d1593bb803726aabd7d0a79b'),('2980836','3527278'),('20261111_180638','20261106_053553')]:
        assert old in source;source=source.replace(old,new)
    needle="terminal['checkpoints'][1]['epoch']==23";assert source.count(needle)==1;source=source.replace(needle,"terminal['checkpoints'][1]['epoch']==21")
    source=source.replace('original-resource Y99','original-resource Y92')
    namespace={'__name__':'y88_original_review','__file__':__file__};exec(compile(ast.parse(source),__file__,'exec'),namespace);return namespace

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('prepare','launch','observe'));parser.add_argument('--token',required=True);parser.add_argument('--label',default='start');a=parser.parse_args();n=module(a.token)
    if a.operation=='observe':n['observe'](a.token,a.label)
    else:n[a.operation](a.token)
