"""CPU source/resource/protocol guards, never actual native efficacy evidence."""
import ast
import copy
import unittest
import yaml
from prepare_y60_stage_heads import ROOT, ref, tick
import y89_deployment as deploy
import advance_y89 as advance


class ExecutionGuards(unittest.TestCase):
    def test_only_original_seven_additions(self):
        self.assertEqual(deploy.git('rev-parse', deploy.Y89 + '^').decode().strip(), deploy.Y88)
        rows = [row.split('\t') for row in deploy.git(
            'diff', '--no-renames', '--name-status', deploy.Y88, deploy.Y89).decode().splitlines()]
        deploy.validate_delta(rows, 'AdaRES-Y89')
        for path in deploy.paths('AdaRES-Y89'):
            self.assertEqual((ROOT / path).read_bytes(), deploy.git('show', deploy.Y89 + ':' + path))

    def test_modified_or_unrelated_source_refused(self):
        rows = [['A', path] for path in deploy.paths('AdaRES-Y89')]
        for bad in (rows[:-1], rows + [['A', 'other.py']], [['M', rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, 'AdaRES-Y89')
        with self.assertRaises(AssertionError):
            deploy.validate_delta(rows, 'AdaRES-Y90')

    def test_active_original_Y88_sources_preserved(self):
        self.assertEqual(set(deploy.contract(deploy.Y88, deploy.Y89, 'AdaRES-Y88')),
                         {deploy.Y88, deploy.Y89})

    def test_original_Y87_source_chain_preserved(self):
        self.assertEqual(set(deploy.contract(deploy.Y87, deploy.Y89, 'AdaRES-Y87')),
                         {deploy.Y87, deploy.Y88, deploy.Y89})

    def test_reversed_unknown_or_divergent_branches_refused(self):
        for commit, head in ((deploy.Y89, deploy.Y88), ('0' * 40, deploy.Y89),
                (deploy.Y88, '9a64d22c41bb8527ca7ca1abba4340546506b1b1')):
            with self.assertRaises(AssertionError):
                deploy.contract(commit, head, 'AdaRES-Y88')

    def test_failed_or_other_prepared_models_not_selected(self):
        for cid in ('AdaRES-Y86', 'AdaRES-Y90', 'AdaRES-Y91'):
            self.assertNotIn(cid, deploy.ADDITIONS.values())

    def test_pause_goal_and_whitelists_unchanged(self):
        servers = yaml.safe_load((ROOT / 'SERVERS.yaml').read_bytes())['servers']
        self.assertEqual(servers['server-1']['allowed_gpus'], [0])
        self.assertEqual(servers['server-2']['allowed_gpus'], [0])
        self.assertEqual(servers['server-6']['allowed_gpus'], [])
        self.assertEqual(ref('pipeline/goal.json')['sha256'],
                         '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a')
        self.assertEqual(ref(advance.GRANT)['sha256'], advance.GRANT_SHA)

    def test_original_native_payload_still_bound(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate, _, _, frozen = advance.prepared(work)
        self.assertEqual(candidate['code_commit'], deploy.Y89)
        self.assertEqual(frozen['runtime_server'], 'server-2')
        self.assertEqual(frozen['runtime_physical_gpu'], 0)
        self.assertEqual(frozen['runtime_power_limit_watts'], 280.)

    def test_unreserved_does_not_treat_pause_as_permission(self):
        pause = dict(id='AdaRES-Y57', status='paused', gpu=1, dispatch_hold=True,
                     resume_requires_user_authorization=True)
        queue = dict(gpu_queues={'server-2': [pause]})
        advance.unreserved(queue, {}, 'server-2', 0)
        for status in ('waiting', 'claiming', 'running'):
            changed = copy.deepcopy(queue)
            changed['gpu_queues']['server-2'].append(dict(id='foreign', status=status, gpu=0))
            with self.assertRaises(AssertionError):
                advance.unreserved(changed, {}, 'server-2', 0)
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, {'server-2': {'0': 'foreign'}}, 'server-2', 0)
        queue['gpu_queues']['server-2'][0]['status'] = 'running'
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, {}, 'server-2', 0)

    def test_physical_busy_foreign_nonfinite_or_power_change_refused(self):
        advance.idle(dict(gpu='0,1,280', compute=''), 280.)
        for gpu, compute in [('nan,0,280', ''), ('0,0,350', ''), ('0,2000,280', ''),
                             ('0,0,280', '99'), ('100,0,280', '')]:
            with self.assertRaises(AssertionError):
                advance.idle(dict(gpu=gpu, compute=compute), 280.)

    def test_entire_original_review_and_training_preserved(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        parentref = next(row for row in work['candidates']
                         if row['id'] == 'AdaRES-Y87')['post_training_freeze']
        parent = tick.load(ROOT / parentref['path'])
        before = copy.deepcopy(parent)
        freeze = advance.make_freeze(parent, parentref)
        self.assertEqual(parent, before)
        self.assertEqual(freeze['training'], parent['training'])
        restored = copy.deepcopy(freeze['evaluation'])
        restored['methods'][0] = 'AdaRES-Y87'
        self.assertEqual(restored, parent['evaluation'])
        self.assertEqual(freeze['parent_protocol'], parentref)
        self.assertEqual(freeze['resource_condition']['power_limit_watts'], 280.)

    def test_wrong_protocol_lineage_or_resource_refused(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        parentref = next(row for row in work['candidates']
                         if row['id'] == 'AdaRES-Y87')['post_training_freeze']
        parent = tick.load(ROOT / parentref['path'])
        for key, value in [('id', 'AdaRES-Y88'), ('code_commit', '0' * 40)]:
            with self.assertRaises(AssertionError):
                advance.make_freeze(dict(parent, **{key: value}), parentref)
        for key, value in [('gpu', 1), ('server', 'server-1'), ('power_limit_watts', 390.)]:
            changed = copy.deepcopy(parent)
            changed['resource_condition'][key] = value
            with self.assertRaises(AssertionError):
                advance.make_freeze(changed, parentref)
        with self.assertRaises(AssertionError):
            advance.make_freeze(parent, dict(parentref, sha256='0' * 64))

    def test_helpers_parse_on_original_python38(self):
        for name in ('advance_y89.py', 'y89_deployment.py', 'test_y89_admission.py'):
            ast.parse((ROOT / 'scripts' / name).read_text(encoding='utf-8'), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
