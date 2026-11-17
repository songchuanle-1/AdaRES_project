"""Describe a failed asset guard; never turn missing/different assets into a PASS."""
import argparse,ast,json
from prepare_y60_stage_heads import ROOT,lease,ref,tick
import check_y101_frozen_review_assets as original

REMOTE='''import datetime,hashlib,json,pathlib,shutil,subprocess,sys,yaml
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
root=pathlib.Path(P['root']);proc=pathlib.Path('/proc/2263581')
assert proc.is_dir() and (proc/'cwd').resolve()==root
assert P['active_config'].encode() in (proc/'cmdline').read_bytes().split(b'\\0')
assert (root/'.codex_source_commit').read_text().strip()=='6fb26984ae652efbba177ab20c39575c44a69517'
assert 'torch' not in sys.modules
sources=[]
for relative,digest in P['sources'].items():
    path=root/relative;present=path.is_file();actual=hashlib.sha256(path.read_bytes()).hexdigest() if present else None
    sources.append(dict(path=relative,present=present,expected_sha256=digest,actual_sha256=actual,matched=present and actual==digest))
configs=[]
for relative,value in P['configs'].items():
    path=root/relative;present=path.is_file();actual=yaml.safe_load(path.read_text()) if present else None
    configs.append(dict(path=relative,present=present,semantic_match=present and actual==value))
weights=[]
for item in P['references']:
    path=pathlib.Path(item['mapped_checkpoint']);exists=path.is_file()
    weights.append(dict(item,exists=exists,bytes=path.stat().st_size if exists else None,checkpoint_loaded=False,weights_sha256_verified=False))
evaluator=root/'tools/evaluate_adares_frozen_full.py';present=evaluator.is_file()
digest=hashlib.sha256(evaluator.read_bytes()).hexdigest() if present else None
assert pathlib.Path(P['data_root']).is_dir() and 'torch' not in sys.modules
print(json.dumps(dict(complete=True,observed_at=datetime.datetime.now().astimezone().isoformat(),hostname='dl02-MS-7B73',references=weights,configs_verified=configs,source_files_verified=sources,all_baseline_sources_match=all(r['matched'] for r in sources),all_native_YAML_semantics_match=all(r['semantic_match'] for r in configs),evaluator_present=present,evaluator_sha256=digest,evaluator_matches=present and digest==P['evaluator_sha256'],original_data_root=P['data_root'],free_bytes=shutil.disk_usage(P['work']).free,torch_imported=False,checkpoint_loaded=False,CUDA_used=False,remote_mutations=False,active_training_preserved=True,source_guard_passed=all(r['matched'] for r in sources))))
'''

def run(token):
    lease(token,300)
    source=ROOT/'scripts/check_y101_frozen_review_assets.py';oldref=ref(source)
    tree=ast.parse(source.read_text(encoding='utf-8'))
    replaced=0
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in node.targets):node.value=ast.Constant(REMOTE);replaced+=1
    assert replaced==1
    class Describe(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):node.value=node.value.replace('y101_review_assets_','y101_review_assets_diagnostic_')
            return node
        def visit_keyword(self,node):
            if node.arg=='baseline_source_and_native_YAML_verified':node.value=ast.parse("raw['all_baseline_sources_match'] and raw['all_native_YAML_semantics_match']",mode='eval').body
            elif node.arg=='readiness':node.value=ast.parse("'blocked_missing_or_different_assets' if not(raw['all_baseline_sources_match'] and raw['all_native_YAML_semantics_match'] and raw['evaluator_matches'] and all(r['exists'] for r in raw['references'])) else 'inventory_only_not_an_executable_evaluation_plan'",mode='eval').body
            return self.generic_visit(node)
    tree=Describe().visit(tree);body=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(body,feature_version=(3,8))
    program=ROOT/('pipeline/checks/y101_review_asset_diagnostic_'+token+'_program.py');assert not program.exists();program.write_text(body,encoding='utf-8',newline='\n')
    namespace={'__name__':'y101_asset_diagnostic','__file__':__file__};exec(compile(body,str(program),'exec'),namespace);namespace['run'](token)
    assert ref(source)==oldref
    work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101')
    out=tick.load(ROOT/candidate['frozen_review_asset_inventory']['path']);raw=tick.load(ROOT/('pipeline/checks/y101_review_assets_diagnostic_'+token+'_remote.json'))
    print(json.dumps(dict(diagnostic=candidate['frozen_review_asset_inventory'],source_guard_passed=raw['source_guard_passed'],nonmatching_sources=[r for r in raw['source_files_verified'] if not r['matched']],nonmatching_configs=[r for r in raw['configs_verified'] if not r['semantic_match']],missing_weights=out['missing_reference_weights'],evaluator_present=raw['evaluator_present'],evaluator_matches=raw['evaluator_matches'],weights_loaded=False,GPU_used=False,original_failed_guard=ref('pipeline/checks/y101_review_assets_'+token+'_remote.log'),program=ref(program))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);run(p.parse_args().token)
