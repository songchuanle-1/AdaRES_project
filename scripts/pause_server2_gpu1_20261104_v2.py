"""Apply GPU1 pause while preserving the observed other user's workload; v1 retained."""
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from prepare_y60_stage_heads import ROOT, monitor, ref, tick

OWNER = '01a07e32-78bf-7670-925b-370911ea48db'
REQUEST = 'server2-gpu1 先暂停使用吧'
AUTH = 'pipeline/authorizations/server2_gpu1_pause_20261104.json'

REMOTE = r'''import json, socket, subprocess
from pathlib import Path
from datetime import datetime, timezone
assert socket.gethostname() == 'dl01-MS-7C60'
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,power.limit', '--format=csv,noheader,nounits']).decode()
compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']).decode()
one = next(v.split(',') for v in gpu.splitlines() if v.split(',')[0].strip() == '1')
assert float(one[4]) == 390
gpu1_pids = [int(v.split(',')[1]) for v in compute.splitlines() if v.split(',')[0].strip() == one[1].strip()]
other_users = {str(pid): subprocess.check_output(['ps','-p',str(pid),'-o','user=']).decode().strip() for pid in gpu1_pids}
assert all(user and user != 'songcl' for user in other_users.values())
assert not Path('/proc/120624/cmdline').exists()
log = Path('/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y81_query_mask_refinement_res_20261103_110717.log')
lines = log.read_text(errors='replace').splitlines()
assert any('Val[30/30]' in v and '363/363' in v for v in lines[-20:])
directory = Path('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y81_query_mask_refinement_res/20261102_220631')
assert any('save last pth to ' + str(directory / 'last.pth') in v for v in lines[-5:])
files = {name: dict(size=(directory/name).stat().st_size, mtime_ns=(directory/name).stat().st_mtime_ns) for name in ('last.pth', 'best.pth')}
assert all(v['size'] > 0 for v in files.values())
print(json.dumps(dict(observed_at=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(), gpu_inventory=gpu, compute=compute, gpu1_idle=not gpu1_pids and float(one[2])<10 and float(one[3])<1024, other_user_compute=other_users, AdaRES_GPU1_compute_present=False, Y81_process_absent=True, Y81_last_lines=lines[-5:], Y81_checkpoint_stats=files, checkpoint_contents_loaded=False, terminal_archive_executed=False, remote_mutations=False)))
'''


def main():
    assert os.environ['CODEX_THREAD_ID'] == OWNER
    assert tick.load(ROOT / 'pipeline/execution_mode.json')['target_thread_id'] == OWNER
    for relative in ('pipeline/headless/.foreground.lockdir', 'pipeline/headless/.tick.lockdir', 'pipeline/headless/.tick_context_state.next.json'):
        assert not (ROOT / relative).exists(), relative
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        auth, proof, observed = [ROOT / p for p in (AUTH, 'pipeline/checks/server2_gpu1_pause_20261104.json', 'pipeline/checks/server2_gpu1_pause_20261104_observation.json')]
        assert not any(p.exists() for p in (auth, proof, observed))
        paths = ['SERVERS.yaml', 'AGENTS.md', 'pipeline/queue.json', 'pipeline/running.json', 'pipeline/research_work.json', 'pipeline/codex_memory.md', 'pipeline/goal.json']
        before = {p: (ROOT / p).read_bytes() for p in paths}
        receipts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'pipeline/admissions').glob('*.json')}
        assert hashlib.sha256(before['pipeline/goal.json']).hexdigest() == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        queue = tick.load(ROOT/'pipeline/queue.json')
        y81 = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y81')
        assert y81['pid'] == 120624 and str(y81['gpu']) == '1'
        assert all(t['id'] in ('AdaRES-Y57', 'AdaRES-Y81') for t in queue['gpu_queues']['server-2'] if str(t.get('gpu')) == '1')
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'songcl@192.168.1.163', 'PYTHONDONTWRITEBYTECODE=1 /home/songcl/.conda/envs/adares_project/bin/python -B -'], input=REMOTE.encode(), capture_output=True, timeout=35)
        assert result.returncode == 0, result.stderr.decode(errors='replace')
        observation = json.loads(result.stdout)
        assert not observation['AdaRES_GPU1_compute_present'] and observation['Y81_process_absent']
        tick.save(observed, observation)
        snapshots = {}
        for name, raw in before.items():
            target = ROOT / ('pipeline/snapshots/before_gpu1_pause_20261104_' + name.replace('/', '_'))
            assert not target.exists()
            target.write_bytes(raw)
            snapshots[name] = ref(target)
        now = datetime.now().astimezone().isoformat()
        tick.save(auth, dict(project='AdaRES_project', source_thread_id=OWNER, human_reply=REQUEST,
            recorded_at=now, server='server-2', host='192.168.1.163', expected_hostname='dl01-MS-7C60',
            resource_use_paused=True, paused_gpu=1, allowed_gpus=[0],
            supersedes_resource_grant=ref('pipeline/authorizations/server2_gpu1_20261029.json'),
            prohibited_until_explicit_user_reauthorization=['GPU1_native_checks', 'GPU1_training_and_dispatch', 'GPU1_frozen_evaluation_and_cost'],
            preserve=['server1_GPU0_350W', 'server2_GPU0_280W', 'Y82', 'Y83', 'Y57_separate_pause', 'all_checkpoints_and_receipts', 'original_frozen_protocol', 'owning_thread_and_schedule'],
            resource_resume_requires_user_authorization=True, Y57_resume_authorized=False,
            observation=ref(observed), hardware_changes=False, scientific_goal_bytes_unchanged=True))
        config = before['SERVERS.yaml'].decode('utf-8')
        old = '# User reauthorized GPU1 on 2026-10-29; Y57 itself remains paused.\n    allowed_gpus: [0, 1]'
        new = '# User paused GPU1 on 2026-11-04; explicit reauthorization required.\n    allowed_gpus: [0]'
        assert config.count(old) == 1
        (ROOT/'SERVERS.yaml').write_text(config.replace(old, new), encoding='utf-8', newline='\n')
        agents = before['AGENTS.md'].decode('utf-8')
        section = '''
## Server-2 GPU1 Paused By User (2026-11-04)

- The owning user said: "server2-gpu1 先暂停使用吧". Stop all AdaRES use of server-2 GPU1 until explicit user reauthorization, including native tests, dispatch, training and frozen evaluation/cost.
- Current permitted resources are server-1 GPU0 at its existing350W and server-2 GPU0 at280W, one independent experiment per GPU. Preserve Y82/Y83. No power or hardware changes.
- This latest pause supersedes the 2026-10-29 GPU1 resource grant and any stale heartbeat text. SERVERS.yaml must keep server-2 allowed_gpus: [0]. Authority: pipeline/authorizations/server2_gpu1_pause_20261104.json.
- GPU1 contains only observed other-user compute, which must remain untouched; Y81 PID120624 was absent and native Train/Val30 completed with last.pth saved. Preserve Y81 artifacts and perform only read-only exact terminal verification/archival; do not restart it or claim complete checkpoint integrity from log/stat checks alone.
- Keep all original GPU1 frozen evaluation/cost resource requirements on hold; do not migrate, retrain or substitute resources/protocols to bypass this pause. Preserve old signed receipts and scientific goal bytes.
- Y57 remains separately paused with its original epoch27 model/optimizer/scheduler; GPU1 reauthorization alone never authorizes Y57 continuation.
- Keep the owning thread, scheduler frequency, data/runtime mappings and server-6 ban. Independent permitted GPU0 work may continue.

'''
        header = '# Current User Resource Instructions\n'
        assert agents.startswith(header)
        (ROOT/'AGENTS.md').write_text(agents.replace(header, header+section, 1), encoding='utf-8', newline='\n')
        y81['dispatch_hold'] = 'User paused server-2 GPU1 on 2026-11-04; no restart/dispatch; completed log awaits exact terminal archival.'
        y81['resource_pause'] = ref(auth)
        tick.save(ROOT/'pipeline/queue.json', queue)
        memory = before['pipeline/codex_memory.md'].decode('utf-8').splitlines()
        at = memory.index('## Goal And Phase') + 1
        memory.insert(at, '- 最新手动资源指令2026-11-04：用户“server2-gpu1 先暂停使用吧”；仅server1GPU0既有350W/server2GPU0保持280W。GPU1测试/派发/训练/完整后评全部暂停，须明确重新授权。'+AUTH+'；覆盖下面所有旧GPU1授权/三卡busy文字。Y57单独暂停保持。')
        at = memory.index('## Running And Queue') + 1
        memory.insert(at, '- 2026-11-04手动核实GPU1仅另一用户compute，无AdaRES compute，保留外部任务；Y81 PID120624缺席、原Val30/30 363/363与last.pth保存结束；未加载checkpoint/未执行完整终态校验或归档，原last/best仅stat保留。Y81登记running属待核归档，不再调用live递增observer；旧record_prepared_y84_y85_wait.py已不适用。Y82/Y83继续，Y84优先Y85仅等授权GPU0；原GPU1计划/冻结后评保留但禁执行。')
        assert len(memory) < 120
        (ROOT/'pipeline/codex_memory.md').write_text('\n'.join(memory)+'\n', encoding='utf-8', newline='\n')
        loaded = monitor.load_servers(str(ROOT))
        assert loaded['server-2']['allowed_gpus'] == loaded['server-1']['allowed_gpus'] == [0]
        assert loaded['server-6']['allowed_gpus'] == []
        for name in ('pipeline/goal.json', 'pipeline/running.json', 'pipeline/research_work.json'):
            assert (ROOT/name).read_bytes() == before[name]
        expected = json.loads(before['pipeline/queue.json'])
        e81 = next(t for t in expected['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y81')
        e81.update(dispatch_hold=y81['dispatch_hold'], resource_pause=ref(auth))
        assert expected == queue
        assert receipts == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'pipeline/admissions').glob('*.json')}
        decision = monitor.log_decision(str(ROOT), 'manual', '用户暂时停用server-2 GPU1',
            'server-2 allowed_gpus收为[0]；GPU1 native/训练/派发/冻结后评暂停，等待明确重新授权；Y82/Y83继续。',
            '用户本会话明确指令；GPU1实核仅另一用户任务，AdaRES Y81自然结束，无停止进程或功耗变更；保留所有原产物和旧收据。', AUTH)
        tick.save(proof, dict(complete=True, observed_at=now, authorization=ref(auth), observation=ref(observed),
            before=snapshots, current_servers=ref('SERVERS.yaml'), current_agents=ref('AGENTS.md'),
            queue=ref('pipeline/queue.json'), memory=ref('pipeline/codex_memory.md'), decision=ref(decision),
            original_receipts_unchanged=True, scientific_goal_bytes_unchanged=True, running_map_unchanged=True,
            research_work_unchanged=True, remote_mutations=False, process_stop_required=False,
            terminal_archive_executed=False, automation_prompt_update_pending=True,
            retained_failed_helper=ref('scripts/pause_server2_gpu1_20261104.py'),
            first_attempt_failed_before_mutations='Idle precondition rejected because GPU1 was taken by another user; second attempt verifies no AdaRES compute and preserves the foreign workload.'))
        print(json.dumps(dict(authorization=ref(auth), proof=ref(proof), allowed_server2_gpus=[0], GPU1_idle=observation['gpu1_idle'], other_user_compute_preserved=True, Y81_process_absent=True)))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
