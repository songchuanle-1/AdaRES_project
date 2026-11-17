"""Collect the existing Y99 eight native passes using unchanged frozen analysis."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,ref,tick
import closure

def collect(token):
    lease(token,600)
    w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y96')
    assert c['code_commit']=='acb8c1769b7f3d66a92d287da8859a06a138b8b0' and '+65536/total164695173/' in c['matched_control']
    p=ast.literal_eval(ast.parse(closure.artifact(ROOT,c['frozen_review_job']['program']).read_text(encoding='utf-8')).body[0].value)
    assert p['code_commit']==c['code_commit'] and p['runtime_marker']=='6fb26984ae652efbba177ab20c39575c44a69517'
    base=ROOT/'scripts/collect_y101_full_review_v2.py';source=base.read_text(encoding='utf-8')
    for old,new in [('AdaRES-Y101','AdaRES-Y96'),('y101_full_review_','y96_full_review_'),('2263581','3893460'),('164695173','164695173')]:
        assert old in source;source=source.replace(old,new)
    source=source.replace('worker_historical_repeat_noise_retained=True','worker_historical_repeat_noise_retained=True,worker_used_original_corrected_repeat_noise=True')
    namespace={'__name__':'y96_collection','__file__':__file__}
    exec(compile(ast.parse(source),__file__,'exec'),namespace)
    namespace['collect'](token)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);collect(parser.parse_args().token)
