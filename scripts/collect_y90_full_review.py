"""Collect the existing original Y94 eight passes; reuse exact frozen coverage/statistics checks."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,tick
import closure

def collect(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y90')
    assert c['code_commit']=='2d2cccb4f6f6ef9daf960b0ac255a3a3ee5e9f71' and '+65536' in c['matched_control']
    p=ast.literal_eval(ast.parse(closure.artifact(ROOT,c['frozen_review_job']['program']).read_text(encoding='utf-8')).body[0].value)
    assert p['code_commit']==c['code_commit'] and p['runtime_marker']=='6fb26984ae652efbba177ab20c39575c44a69517'
    source=(ROOT/'scripts/collect_y101_full_review_v2.py').read_text(encoding='utf-8')
    for old,new in [('AdaRES-Y101','AdaRES-Y90'),('y101_full_review_','y90_full_review_'),('2263581','2442962'),('164695173','164695173')]:
        assert old in source;source=source.replace(old,new)
    source=source.replace('worker_historical_repeat_noise_retained=True','worker_historical_repeat_noise_retained=True,worker_used_original_corrected_repeat_noise=True')
    namespace={'__name__':'y90_collection','__file__':__file__};exec(compile(ast.parse(source),__file__,'exec'),namespace);namespace['collect'](token)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);collect(parser.parse_args().token)
