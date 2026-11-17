P={'hostname': 'dl01-MS-7C60', 'pid': 52561, 'log': '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y70_query_scale_fusion_res_20261030_184633.log'}
import json,pathlib,subprocess,re,datetime
assert subprocess.check_output(['hostname']).decode().strip()==P['hostname']
p=pathlib.Path('/proc')/str(P['pid']);f=pathlib.Path(P['log'])
with f.open('rb') as r:r.seek(max(0,f.stat().st_size-24000));tail=r.read().decode(errors='replace')
rows=re.findall(r'(Train|Val)\[(\d+)/(\d+)\]:[^\r\n]*?(\d+)/(\d+)\s*\[',tail)
print(json.dumps(dict(complete=True,pid=P['pid'],process_live=p.exists(),last_phase=rows[-1] if rows else None,tail=tail[-12000:],observed_at=datetime.datetime.now().astimezone().isoformat())))
