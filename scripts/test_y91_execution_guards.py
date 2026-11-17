"""Side-effect-free source, protocol and resource guards for retained Y91."""
import ast
import copy
import unittest
import yaml
from prepare_y60_stage_heads import ROOT, monitor, ref, tick
from check_ef_execution import validate_frozen
from run_y91_bound_native import runtime, validate_terminal
import advance_y91 as advance
import y91_deployment as deploy

class ExecutionGuards(unittest.TestCase):

    def candidate(self):
        return next((row for row in tick.load(ROOT / 'pipeline/research_work.json')['candidates'] if row['id'] == 'AdaRES-Y91'))

    def parent(self):
        row = next((row for row in tick.load(ROOT / 'pipeline/research_work.json')['candidates'] if row['id'] == 'AdaRES-Y89'))
        return (tick.load(ROOT / row['post_training_freeze']['path']), row['post_training_freeze'])

    def test_only_retained_seven_additions(self):
        self.assertEqual(deploy.git('rev-parse', deploy.Y91 + '^').decode().strip(), deploy.Y90)
        rows = [row.split('\t') for row in deploy.git('diff', '--no-renames', '--name-status', deploy.Y90, deploy.Y91).decode().splitlines()]
        deploy.validate_delta(rows, 'AdaRES-Y91')
        for path in deploy.paths('AdaRES-Y91'):
            self.assertEqual((ROOT / path).read_bytes(), deploy.git('show', deploy.Y91 + ':' + path))

    def test_modified_or_unrelated_source_refused(self):
        rows = [['A', path] for path in deploy.paths('AdaRES-Y91')]
        for bad in (rows[:-1], rows + [['A', 'other.py']], [['M', rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, 'AdaRES-Y91')
        with self.assertRaises(AssertionError):
            deploy.validate_delta(rows, 'AdaRES-Y91')

    def test_active_Y89_sources_and_original_marker_preserved(self):
        self.assertEqual(set(deploy.contract(deploy.Y90, deploy.Y91, 'AdaRES-Y90')), {deploy.Y90, deploy.Y91})

    def test_original_Y88_source_chain_preserved(self):
        self.assertEqual(set(deploy.contract(deploy.Y89, deploy.Y91, 'AdaRES-Y89')), {deploy.Y89, deploy.Y90, deploy.Y91})

    def test_reversed_unknown_or_divergent_branches_refused(self):
        for commit, head in ((deploy.Y91, deploy.Y90), ('0' * 40, deploy.Y91), (deploy.Y90, '9a64d22c41bb8527ca7ca1abba4340546506b1b1')):
            with self.assertRaises(AssertionError):
                deploy.contract(commit, head, 'AdaRES-Y90')
        for cid in ('AdaRES-Y86', 'AdaRES-Y92'):
            self.assertNotIn(cid, deploy.ADDITIONS.values())

    def test_pause_goal_and_whitelists_unchanged(self):
        servers = yaml.safe_load((ROOT / 'SERVERS.yaml').read_bytes())['servers']
        self.assertEqual(servers['server-1']['allowed_gpus'], [0])
        self.assertEqual(servers['server-2']['allowed_gpus'], [0])
        self.assertEqual(servers['server-6']['allowed_gpus'], [])
        self.assertEqual(ref('pipeline/goal.json')['sha256'], '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a')
        self.assertEqual(ref(advance.GRANT)['sha256'], advance.GRANT_SHA)
        self.assertEqual(ref(advance.PAUSE)['sha256'], advance.PAUSE_SHA)

    def test_both_native_payloads_remain_bound(self):
        candidate = self.candidate()
        for name, key in (('server-1', 'native_smoke_plan_server1'), ('server-2', 'native_smoke_plan_server2_gpu0')):
            plan = tick.load(ROOT / candidate[key]['path'])
            program = ROOT / plan['program']['path']
            self.assertEqual(ref(program), plan['program'])
            frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
            self.assertEqual(validate_frozen(candidate, plan, frozen), 7)
            runtime(name, candidate, frozen, monitor.load_servers(str(ROOT))[name])

    def test_live_or_incomplete_original_owner_never_unblocks(self):
        proof = dict(id='AdaRES-Y89', complete=True, process_absent=True, CUDA_initialized=False, completed_epochs={phase: list(range(1, 31)) for phase in ('Train', 'Val')}, checkpoints=[dict(epoch=epoch, scheduler_last_epoch=epoch, optimizer_state_count=1, model_finite=True, optimizer_finite=True) for epoch in (30, 21)])
        validate_terminal(proof, 'server-2')
        for key, value in (('process_absent', False), ('CUDA_initialized', True), ('id', 'AdaRES-Y90')):
            with self.assertRaises(AssertionError):
                validate_terminal(dict(proof, **{key: value}), 'server-2')
        changed = copy.deepcopy(proof)
        changed['checkpoints'][0]['epoch'] = 29
        with self.assertRaises(AssertionError):
            validate_terminal(changed, 'server-2')
        changed = copy.deepcopy(proof)
        changed['completed_epochs']['Val'].pop()
        with self.assertRaises(AssertionError):
            validate_terminal(changed, 'server-2')

    def test_unreserved_does_not_treat_pause_as_permission(self):
        pause = dict(id='AdaRES-Y57', status='paused', gpu=1, dispatch_hold=True, resume_requires_user_authorization=True)
        queue = dict(gpu_queues={'server-1': [], 'server-2': [pause]})
        for name in ('server-1', 'server-2'):
            advance.unreserved(queue, {}, name, 0)
            for status in ('waiting', 'claiming', 'running'):
                changed = copy.deepcopy(queue)
                changed['gpu_queues'][name].append(dict(id='foreign', status=status, gpu=0))
                with self.assertRaises(AssertionError):
                    advance.unreserved(changed, {}, name, 0)
            with self.assertRaises(AssertionError):
                advance.unreserved(queue, {name: {'0': 'foreign'}}, name, 0)

    def test_physical_busy_foreign_nonfinite_or_power_change_refused(self):
        advance.idle(dict(gpu='0,1,280', compute=''), 280.0)
        for gpu, compute in [('nan,0,280', ''), ('0,0,350', ''), ('0,2000,280', ''), ('0,0,280', '99'), ('100,0,280', '')]:
            with self.assertRaises(AssertionError):
                advance.idle(dict(gpu=gpu, compute=compute), 280.0)

    def test_entire_original_review_and_training_preserved(self):
        parent, parentref = self.parent()
        before = copy.deepcopy(parent)
        freeze = advance.make_freeze(parent, parentref)
        self.assertEqual(parent, before)
        self.assertEqual(freeze['training'], parent['training'])
        restored = copy.deepcopy(freeze['evaluation'])
        restored['methods'][0] = 'AdaRES-Y89'
        self.assertEqual(restored, parent['evaluation'])
        self.assertEqual(freeze['parent_protocol'], parentref)
        self.assertEqual(freeze['resource_condition']['power_limit_watts'], 280.0)

    def test_wrong_protocol_lineage_or_resource_refused(self):
        parent, parentref = self.parent()
        for key, value in [('id', 'AdaRES-Y90'), ('code_commit', '0' * 40)]:
            with self.assertRaises(AssertionError):
                advance.make_freeze(dict(parent, **{key: value}), parentref)
        for key, value in [('gpu', 1), ('server', 'server-1'), ('power_limit_watts', 390.0)]:
            changed = copy.deepcopy(parent)
            changed['resource_condition'][key] = value
            with self.assertRaises(AssertionError):
                advance.make_freeze(changed, parentref)
        with self.assertRaises(AssertionError):
            advance.make_freeze(parent, dict(parentref, sha256='0' * 64))

    def test_helpers_parse_on_original_python38(self):
        for name in ('advance_y91.py', 'y91_deployment.py', 'test_y91_admission.py'):
            ast.parse((ROOT / 'scripts' / name).read_text(encoding='utf-8'), feature_version=(3, 8))
if __name__ == '__main__':
    unittest.main()
