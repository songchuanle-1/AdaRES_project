"""Read-only GPU1 occupation after Y26 exclusive diagnostics."""

from datetime import datetime, timezone
from pathlib import Path

gpu = subprocess.check_output(
    ["nvidia-smi", "-i", "1", "--query-gpu=uuid,memory.used,utilization.gpu",
     "--format=csv,noheader,nounits"]).decode().strip().split(",")
apps = subprocess.check_output(
    ["nvidia-smi", "-i", "1", "--query-compute-apps=gpu_uuid,pid,process_name",
     "--format=csv,noheader,nounits"]).decode().strip()
records = []
for line in apps.splitlines():
    fields = [v.strip() for v in line.split(",", 2)]
    if fields[0] == gpu[0].strip():
        records.append(dict(pid=int(fields[1]), name=fields[2]))
old_pid_present = Path("/proc/98611").exists()
result = dict(complete=True, gpu=1, uuid=gpu[0].strip(),
              memory_mib=int(gpu[1]), utilization=int(gpu[2]),
              compute=records, original_y25_pid_present=old_pid_present,
              observed_remote_utc=datetime.now(timezone.utc).isoformat(),
              note="GPU0 not queried/used; original PID absence is not a lifecycle mutation.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
