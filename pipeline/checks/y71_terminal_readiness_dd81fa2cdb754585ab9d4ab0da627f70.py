P={'pid': 53714, 'log': '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y71_visual_product_fusion_res_20261031_071201.log'}
import json,pathlib,subprocess,re,datetime
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
p=pathlib.Path('/proc')/str(P['pid']);log=pathlib.Path(P['log'])
with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-24000));tail=stream.read().decode(errors='replace')
rows=re.findall(r'(Train|Val)\[(\d+)/(\d+)\]:[^\r\n]*?(\d+)/(\d+)\s*\[',tail)
print(json.dumps(dict(complete=True,process_live=p.exists(),last_phase=rows[-1] if rows else None,tail=tail[-12000:],observed_at=datetime.datetime.now().astimezone().isoformat())))
