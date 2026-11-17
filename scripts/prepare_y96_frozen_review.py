"""Prepare the archived Y96 original-resource review; no GPU or launch here."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,tick
import closure

def prepare(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y96')
    assert c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' and c['required_server']=='server-1' and c['required_gpu']==0
    assert c['code_commit']=='acb8c1769b7f3d66a92d287da8859a06a138b8b0' and not c.get('frozen_review_job')
    t=tick.load(closure.artifact(ROOT,c['terminal_evidence']));assert t['checkpoints'][1]['epoch']==23 and '/20261110_165702/' in t['checkpoints'][1]['path']
    closure.artifact(ROOT,c['terminal_archive_binding'])
    base=ROOT/'scripts/y99_frozen_review_job.py';source=base.read_text(encoding='utf-8')
    for old,new in [('AdaRES-Y99','AdaRES-Y96'),('adares_y99_spatial_mask_skip','adares_y96_seed_query_skip'),('y99_','y96_'),('53f5b028bdf24e8b41dd934931d1fedc99843d8d','acb8c1769b7f3d66a92d287da8859a06a138b8b0'),('2980836','3893460'),('20261111_180638','20261110_165702')]:
        assert old in source;source=source.replace(old,new)
    namespace={'__name__':'y96_review_preparation','__file__':__file__}
    exec(compile(ast.parse(source),__file__,'exec'),namespace);namespace['prepare'](token)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);prepare(parser.parse_args().token)
