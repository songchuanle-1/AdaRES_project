"""Record observed Y96 inconclusive science only after original cost and postaudit."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,tick
import closure

def complete(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y96')
    f=tick.load(closure.artifact(ROOT,c['full_frozen_evaluation_result']));assert f['summary']['AdaRES-Y96']['miou']==0.5082609879270078 and not f['target_reached'] and not f['frozen_material_subgroup_harm']
    assert all(g['all']['scene_clustered_ci95'][0]<=0<=g['all']['scene_clustered_ci95'][1] for g in f['comparisons'].values())
    t=tick.load(closure.artifact(ROOT,c['terminal_evidence']));assert t['checkpoints'][1]['epoch']==23
    source=(ROOT/'scripts/complete_y101_scientific_review.py').read_text(encoding='utf-8')
    for old,new in [('AdaRES-Y101','AdaRES-Y96'),('y101_trained_cost_','y96_trained_cost_'),('best24 original checkpoint','best23 original checkpoint')]:
        assert old in source;source=source.replace(old,new)
    namespace={'__name__':'y96_completion','__file__':__file__};exec(compile(ast.parse(source),__file__,'exec'),namespace);namespace['complete'](token)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);complete(parser.parse_args().token)
