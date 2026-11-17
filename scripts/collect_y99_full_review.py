"""Collect the existing Y99 eight native passes using unchanged frozen analysis."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,ref,tick
import closure

def collect(token):
    lease(token,600)
    w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y99')
    assert c['code_commit']=='53f5b028bdf24e8b41dd934931d1fedc99843d8d' and '+8192/total164637829/' in c['matched_control']
    p=ast.literal_eval(ast.parse(closure.artifact(ROOT,c['frozen_review_job']['program']).read_text(encoding='utf-8')).body[0].value)
    assert p['code_commit']==c['code_commit'] and p['runtime_marker']=='6fb26984ae652efbba177ab20c39575c44a69517'
    base=ROOT/'scripts/collect_y101_full_review_v2.py';source=base.read_text(encoding='utf-8')
    for old,new in [('AdaRES-Y101','AdaRES-Y99'),('y101_full_review_','y99_full_review_'),('2263581','2980836'),('164695173','164637829')]:
        assert old in source;source=source.replace(old,new)
    source=source.replace('worker_historical_repeat_noise_retained=True','worker_historical_repeat_noise_retained=True,worker_used_original_corrected_repeat_noise=True')
    namespace={'__name__':'y99_collection','__file__':__file__}
    exec(compile(ast.parse(source),__file__,'exec'),namespace)
    namespace['collect'](token)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);collect(parser.parse_args().token)
