"""Collect Y88's hash-reused candidate and only newly executed original baseline passes."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,tick
import closure

def collect(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y88')
    assert c['code_commit']=='cb4a5813755d38b3d1593bb803726aabd7d0a79b' and '+65536' in c['matched_control']
    job=c['frozen_review_job'];assert job['attempt']==2 and job['reused_completed_methods']==['AdaRES-Y88']
    p=ast.literal_eval(ast.parse(closure.artifact(ROOT,job['program']).read_text(encoding='utf-8')).body[0].value);partial=tick.load(closure.artifact(ROOT,c['partial_frozen_review_capture']))
    assert p['resume_captures']['AdaRES-Y88']['capture_sha256']==partial['completed_capture_verified']['sha256'] and p['code_commit']==c['code_commit'] and p['runtime_marker']=='6fb26984ae652efbba177ab20c39575c44a69517'
    source=(ROOT/'scripts/collect_y90_full_review.py').read_text(encoding='utf-8')
    for old,new in [('AdaRES-Y90','AdaRES-Y88'),('y90_','y88_'),('2d2cccb4f6f6ef9daf960b0ac255a3a3ee5e9f71','cb4a5813755d38b3d1593bb803726aabd7d0a79b'),('2442962','3527278')]:
        assert old in source;source=source.replace(old,new)
    namespace={'__name__':'y88_collection','__file__':__file__};exec(compile(ast.parse(source),__file__,'exec'),namespace);namespace['collect'](token)
    updated=next(x for x in tick.load(ROOT/'pipeline/research_work.json')['candidates'] if x['id']=='AdaRES-Y88');root=ROOT/updated['frozen_review_capture_directory'];assert (root/'adares-y88.json').read_bytes()==closure.artifact(ROOT,partial['completed_capture_verified']).read_bytes()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);collect(parser.parse_args().token)
