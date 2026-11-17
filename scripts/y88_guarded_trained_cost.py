"""Use the unchanged frozen Y88 cost core with the already verified original-device resource sampling guard."""
import argparse,ast,hashlib
from prepare_y60_stage_heads import ROOT,lease,tick,ref
from y90_cost_resource_recovery import GUARD
import closure

GUARD_TEST='pipeline/checks/y90_cost_resource_recovery_06beafa3cffe4ecf956270d1146f75c9.json'

def module(token):
    lease(token,600);guard_ref=ref(GUARD_TEST);guard=tick.load(closure.artifact(ROOT,guard_ref));assert guard['safety_guard_boundary_checks_passed']==8 and guard['original_scientific_core_byte_identical']
    source=(ROOT/'scripts/y90_frozen_trained_cost.py').read_text(encoding='utf-8')
    for old,new in [('AdaRES-Y90','AdaRES-Y88'),('y90_','y88_'),('adares_y88_query_peer_affinity','adares_y88_mask_query_quadratic'),('2d2cccb4f6f6ef9daf960b0ac255a3a3ee5e9f71','cb4a5813755d38b3d1593bb803726aabd7d0a79b'),('2442962','3527278')]:
        assert old in source;source=source.replace(old,new)
    # y90_ replacement already specializes the package prefix; finish its existing suffix explicitly.
    source=source.replace('adares_y88_query_peer_affinity','adares_y88_mask_query_quadratic')
    namespace={'__name__':'y88_guarded_cost_module','__file__':__file__};exec(compile(ast.parse(source),__file__,'exec'),namespace);m=namespace['module'](token)
    header=m['HEADER'];assert header.count('COUNT=0\n')==1;header=header.replace('COUNT=0\n','COUNT=0\n'+GUARD+'\n')
    needle='    COUNT+=1;value=';assert header.count(needle)==1;header=header.replace(needle,'    resource_check()\n    COUNT+=1;value=')
    needle="    root=pathlib.Path(P['root']);source_manifest=P['candidate_source_manifest']";assert header.count(needle)==1;header=header.replace(needle,"    resource_sample();resource_check();GUARD_THREAD=threading.Thread(target=resource_watch,daemon=True);GUARD_THREAD.start()\n"+needle)
    needle="    audit=dict(complete=True,id='AdaRES-Y88'";assert header.count(needle)==1;header=header.replace(needle,"    GUARD_STOP.set();GUARD_THREAD.join(5);assert not GUARD_THREAD.is_alive();resource_sample();resource_check();assert GUARD_SAMPLES>=4 and GUARD_OWNED_SAMPLES>0\n"+needle)
    needle='source_raw_bytes_preserved=True,source_files=after_sources';assert header.count(needle)==1;header=header.replace(needle,"source_raw_bytes_preserved=True,resource_guard_samples_complete=True,resource_guard_samples=GUARD_SAMPLES,resource_guard_owned_samples=GUARD_OWNED_SAMPLES,no_unowned_GPU0_compute_observed=True,resource_guard_log_sha256=sha(folder/'resource_guard.jsonl'),source_files=after_sources")
    m['HEADER']=header
    # Extend only the original build receipt; its AST inverse-equivalence proof and scientific core stay unchanged.
    tree=ast.parse((ROOT/'scripts/y101_frozen_trained_cost.py').read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build')
    text=ast.unparse(node)
    for old,new in [('AdaRES-Y101','AdaRES-Y88'),('y101_','y88_'),('adares_y88_query_ffn_skip','adares_y88_mask_query_quadratic')]:text=text.replace(old,new)
    for key in ('frozen_review_reference_stage','frozen_review_pointnet_origin','frozen_review_runtime_binding'):text=text.replace("c['"+key+"']","shared['"+key+"']")
    needle="code_commit=p['code_commit'], gpu=0";assert text.count(needle)==1;text=text.replace(needle,"code_commit=p['code_commit'], runtime_marker=p['runtime_marker'], gpu=0")
    needle='launch_started=False, checkpoint_loaded=False, GPU_used=False';assert text.count(needle)==1;text=text.replace(needle,"launch_started=False, checkpoint_loaded=False, GPU_used=False, resource_guard_header_instrumented=True, resource_guard_sha256=GUARD_SHA, original_resource_guard_boundary_evidence=GUARD_REF, guard_boundary_checks_reused_not_replayed=8")
    m.update(GUARD_SHA=hashlib.sha256(GUARD.encode()).hexdigest(),GUARD_REF=guard_ref);exec(compile(ast.parse(text),__file__,'exec'),m)
    return m

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('build','launch','observe'));parser.add_argument('--token',required=True);parser.add_argument('--label',default='start');a=parser.parse_args();m=module(a.token)
    if a.operation=='observe':m['observe'](a.token,a.label)
    else:m[a.operation](a.token)
