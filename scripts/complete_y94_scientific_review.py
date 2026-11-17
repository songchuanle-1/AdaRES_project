"""Apply the original prespecified Y94 subgroup-harm criterion after full cost and postaudit."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,tick
import closure

def complete(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y94')
    f=tick.load(closure.artifact(ROOT,c['full_frozen_evaluation_result']));freeze=tick.load(closure.artifact(ROOT,c['post_training_freeze']));t=tick.load(closure.artifact(ROOT,c['terminal_evidence']))
    assert freeze['evaluation']['material_harm']=='Prespecified target-size subgroup difference CI upper<0 versus Y36/W2; retain negatives.' and t['checkpoints'][1]['epoch']==21
    expected=dict(reference='AdaRES-W2',group='small',delta=-0.01673754047576701,ci95=[-0.03112226172233613,-0.0024482440385502685])
    assert f['summary']['AdaRES-Y94']['miou']==0.5075611062753286 and f['frozen_material_subgroup_harm']==[expected] and not f['target_reached']
    small=f['comparisons']['AdaRES-W2']['small'];assert small['frozen_material_harm_ci_upper_below_zero'] and small['material_harm_within_observed_noise'] and small['observed_group_mean_absolute_repeat_difference']==0
    assert all(g['all']['scene_clustered_ci95'][0]<=0<=g['all']['scene_clustered_ci95'][1] for g in f['comparisons'].values())
    source=(ROOT/'scripts/complete_y101_scientific_review.py').read_text(encoding='utf-8')
    replacements=[('AdaRES-Y101','AdaRES-Y94'),('y101_trained_cost_','y94_trained_cost_'),('164695173','164629661'),('best24 original checkpoint','best21 original checkpoint'),('assert not harm and first<.515','assert harm and first<.515'),("'complete_inconclusive'","'rejected_material_subgroup_harm'"),('subgroup_material_harm=False',"subgroup_material_harm=True,frozen_material_subgroup_harm=harm,rejection_scope='This fixed30/best21/seed1999 trained implementation; not a mechanism-wide claim'"),("c.update(stage='complete',","c.update(stage='rejected',failure_scope='implementation',rejection_reason='Original prespecified small-target subgroup CI upper below zero versus W2; one trained realization, no mechanism-wide falsification',"),('Point estimates below Y36/W2 and all overall paired CIs cross zero; no demonstrated improvement or equivalence, and no prespecified subgroup harm.','Point estimates below Y36/W2 and overall paired CIs cross zero. Small-target subgroup versus W2 violates the original frozen CI-upper-below-zero harm criterion; reject this trained realization, not the whole mechanism.'),('Retain inconclusive result and all negatives;','Retain the original frozen subgroup-harm rejection and all negative evidence; reopening this rejected ID requires explicit authorization;'),('subgroup_harm=False','subgroup_harm=True')]
    for old,new in replacements:
        assert old in source,old;source=source.replace(old,new)
    namespace={'__name__':'y94_completion','__file__':__file__};exec(compile(ast.parse(source),__file__,'exec'),namespace);namespace['complete'](token)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);complete(parser.parse_args().token)
