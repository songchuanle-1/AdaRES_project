"""Apply the owning user's GPU1 resource grant before acquiring a new snapshot."""
import hashlib
import json
import os
from datetime import datetime
from prepare_y60_stage_heads import ROOT, monitor, ref, tick

OWNER = '01a07e32-78bf-7670-925b-370911ea48db'
REQUEST = 'server2 GPU1也可以使用'


def main():
    assert os.environ['CODEX_THREAD_ID'] == OWNER
    assert tick.load(ROOT / 'pipeline/execution_mode.json')['target_thread_id'] == OWNER
    assert not (ROOT / 'pipeline/headless/.foreground.lockdir').exists()
    assert not (ROOT / 'pipeline/headless/.tick.lockdir').exists()
    assert not (ROOT / 'pipeline/headless/.tick_context_state.next.json').exists()
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try:
        auth = ROOT / 'pipeline/authorizations/server2_gpu1_20261029.json'
        proof = ROOT / 'pipeline/checks/server2_gpu1_resource_update_20261029.json'
        assert not auth.exists() and not proof.exists()
        config = ROOT / 'SERVERS.yaml'; before = config.read_bytes()
        goal = ROOT / 'pipeline/goal.json'; original_goal = goal.read_bytes()
        goal_sha = hashlib.sha256(original_goal).hexdigest()
        assert goal_sha == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        old_sha = hashlib.sha256(before).hexdigest()
        snapshot = ROOT / ('pipeline/snapshots/SERVERS_' + old_sha + '.yaml')
        if snapshot.exists(): assert snapshot.read_bytes() == before
        else: snapshot.write_bytes(before)
        goal_snapshot = ROOT / ('pipeline/snapshots/goal_' + goal_sha + '.json')
        if goal_snapshot.exists(): assert goal_snapshot.read_bytes() == original_goal
        else: goal_snapshot.write_bytes(original_goal)
        text = before.decode('utf-8')
        old = '# User-authorized temporary GPU0-only trial; Y57/GPU1 is paused.\n    allowed_gpus: [0]'
        new = '# User reauthorized GPU1 on 2026-10-29; Y57 itself remains paused.\n    allowed_gpus: [0, 1]'
        normalized = text.replace('\r\n', '\n')
        assert normalized.count(old) == 1
        tick.save(auth, dict(project='AdaRES_project', authorization_granted=True,
            recorded_on='2026-10-29', source_thread_id=OWNER, human_reply=REQUEST,
            server='server-2', host='192.168.1.163', expected_hostname='dl01-MS-7C60',
            allowed_gpus=[0, 1], newly_authorized_gpu=1, max_tasks_per_gpu=1,
            operations=['native_checks', 'admission_queue_and_training', 'original_frozen_evaluation'],
            preserve=['server1_gpu0_authorization', 'server2_gpu0_280W', 'all_existing_runs',
                'Y57_separate_user_pause_and_checkpoint_resume_requirements', 'original_data',
                'seed1999_fixed30', 'no_server6', 'no_scheduler_frequency_change'],
            Y57_resume_authorized=False, hardware_changes_authorized=False,
            goal_resource_sentence_override='The resource-only GPU0/GPU1 pause sentence in the '
                'frozen goal is superseded for GPU1 availability by this explicit grant. '
                'Y57 itself remains paused; scientific goal/data/protocol and existing receipt hashes stay exact.',
            frozen_scientific_goal_sha256=goal_sha))
        config.write_text(normalized.replace(old, new), encoding='utf-8', newline='\n')
        agents = ROOT / 'AGENTS.md'; content = agents.read_text(encoding='utf-8')
        section = '''## Server-2 GPU1 Reauthorized By User (2026-10-29)

- The owning user said: "server2 GPU1也可以使用".
- AdaRES may use server-2 GPU0 and GPU1 plus server-1 GPU0, one independent experiment per GPU.
- Keep server-2 GPU0 at280W. GPU1 was observed at its existing390W limit; no power/hardware change is authorized or needed.
- This resource grant supersedes earlier GPU1/GPU0-only resource restrictions, including the resource-only sentence in the frozen goal and old heartbeat text.
- Y57's separate experiment pause remains in effect. Preserve its original epoch27 model/optimizer/scheduler checkpoint; resume only on an explicit Y57 continuation request.
- Reuse the existing verified private runtime and exact dataset. Preserve Y65/Y66, foreign jobs, all scientific/data/seed/fixed30 boundaries and server-6 ban.
- The scientific goal file stays byte-identical to preserve signed running receipts; current resource authority is this section, SERVERS.yaml and pipeline/authorizations/server2_gpu1_20261029.json.
- Keep the existing owning thread and schedule frequency; update only the saved resource instructions to match this grant.

'''
        assert content.startswith('# Current User Resource Instructions\n')
        agents.write_text(content.replace('# Current User Resource Instructions\n',
            '# Current User Resource Instructions\n\n' + section, 1), encoding='utf-8', newline='\n')
        loaded = monitor.load_servers(str(ROOT))
        assert loaded['server-2']['allowed_gpus'] == [0, 1]
        assert loaded['server-1']['allowed_gpus'] == [0] and loaded['server-6']['allowed_gpus'] == []
        assert goal.read_bytes() == original_goal
        decision = monitor.log_decision(str(ROOT), 'manual', '用户重新授权server-2 GPU1',
            'GPU1加入可用资源，优先推进已准备的Y67；Y57单独暂停仍保留，Y65/Y66原运行不变。',
            '本会话明确授权，资源上限每卡一项；GPU0保持280W，GPU1既有390W不改。科学goal字节不改以保留原运行收据。',
            ref(auth)['path'])
        tick.save(proof, dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
            authorization=ref(auth), original_server_configuration=ref(snapshot),
            current_server_configuration=ref(config), goal=ref(goal_snapshot),
            scientific_goal_bytes_unchanged=True, current_agents=ref(agents), decision=ref(decision),
            queue_changed=False, running_changed=False, remote_mutations=False, Y57_still_paused=True))
        print(json.dumps(dict(authorization=ref(auth), proof=ref(proof), allowed_server2_gpus=[0, 1])))
    finally: lock.close()


if __name__ == '__main__': main()
