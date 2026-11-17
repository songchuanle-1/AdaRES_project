P={'hostname': 'dl02-MS-7B73', 'pid': 1135633, 'log': '/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y69_coarse_geometry_bias_res_20261030_131753.log'}
import json,pathlib,subprocess,re,datetime
assert subprocess.check_output(['hostname']).decode().strip()==P['hostname']
p=pathlib.Path('/proc')/str(P['pid']);f=pathlib.Path(P['log'])
with f.open('rb') as r:r.seek(max(0,f.stat().st_size-24000));tail=r.read().decode(errors='replace')
rows=re.findall(r'(Train|Val)\[(\d+)/(\d+)\]:[^
]*?(\d+)/(\d+)\s*\[',tail)
print(json.dumps(dict(complete=True,pid=P['pid'],process_live=p.exists(),last_phase=rows[-1] if rows else None,tail=tail[-12000:],observed_at=datetime.datetime.now().astimezone().isoformat())))
