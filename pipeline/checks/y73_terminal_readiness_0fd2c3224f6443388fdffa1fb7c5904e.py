P={'pid': 62829, 'root': '/bak_home/songcl/Project/AdaRES_project/AdaRES', 'config': 'configs_song/model_paper5/adares_y73_initial_query_anchor_res.yaml', 'log': '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y73_initial_query_anchor_res_20261031_171158.log'}
import datetime,json,pathlib,re,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
proc=pathlib.Path('/proc')/str(P['pid']);live=proc.is_dir()
if live:
 assert (proc/'cwd').resolve()==pathlib.Path(P['root'])
 args=(proc/'cmdline').read_bytes().split(b'\0')
 assert P['config'].encode() in args and b'tools/train.py' in args
log=pathlib.Path(P['log'])
with log.open('rb') as stream:
 stream.seek(max(0,log.stat().st_size-32000));tail=stream.read().decode(errors='replace')
rows=re.findall(r'(Train|Val)\[(\d+)/30\]:[^\r\n]*?(\d+)/(4686|363)\s*\[',tail)
assert rows
assert not any(v in tail for v in ('Traceback','CUDA out of memory','RuntimeError'))
print(json.dumps(dict(complete=True,process_live=live,pid=P['pid'],last_phase=rows[-1],log_bytes=log.stat().st_size,observed_at=datetime.datetime.now().astimezone().isoformat(),checkpoint_loaded=False,remote_mutations=False)))
