"""Actual retained-source/runtime tests; no fabricated native success."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick, monitor, ref
from check_ef_execution import validate_frozen
from prepare_y69_server1_alternative import ADAPT
from run_y98_native import runtime, validate_terminal, validate_binding, PARENT

class RuntimePlan(unittest.TestCase):

    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y98'))
        self.servers = monitor.load_servers(str(ROOT))
        self.values = {}
        for name, key in (('server-1', 'native_smoke_plan_server1'), ('server-2', 'native_smoke_plan_server2_gpu0')):
            plan = tick.load(ROOT / self.c[key]['path'])
            source = (ROOT / plan['program']['path']).read_text(encoding='utf-8')
            self.values[name] = (plan, ast.literal_eval(ast.parse(source).body[0].value), source.partition('\n')[2])

    def test_seven_tested_files_and_runtimes(self):
        for name, (plan, frozen, _) in self.values.items():
            self.assertEqual(validate_frozen(self.c, plan, frozen), 7)
            self.assertEqual(runtime(name, self.c, frozen, self.servers[name]), 350.0 if name == 'server-1' else 280.0)

    def test_scientific_payload_identical(self):
        a, b = [value[1] for value in self.values.values()]
        for key in ('variant', 'code_commit', 'goal_sha256', 'train_source_lf_sha256'):
            self.assertEqual(a[key], b[key])

    def test_only_existing_path_and_hostname_adaptation(self):
        first, second = (self.values['server-1'][2], self.values['server-2'][2])
        self.assertEqual(first.replace(ADAPT, '').replace("== 'dl02-MS-7B73'", "== 'dl01-MS-7C60'"), second)

    def test_GPU1_power_host_and_runtime_changes_refused(self):
        for name, (_, frozen, _) in self.values.items():
            for key, value in (('runtime_physical_gpu', 1), ('runtime_power_limit_watts', 390.0), ('runtime_server', 'server-6')):
                with self.subTest(name=name, key=key), self.assertRaises(AssertionError):
                    runtime(name, self.c, dict(frozen, **{key: value}), self.servers[name])
            for key, value in (('expected_hostname', 'foreign'), ('python', '/wrong/python'), ('allowed_gpus', [0, 1])):
                with self.subTest(name=name, key=key), self.assertRaises(AssertionError):
                    runtime(name, self.c, frozen, dict(self.servers[name], **{key: value}))

    def test_stale_goal_tampered_source_and_wrong_commit_refused(self):
        for plan, frozen, _ in self.values.values():
            for field in ('goal', 'source', 'commit'):
                changed = copy.deepcopy(frozen)
                if field == 'goal':
                    changed['goal_sha256'] = '0' * 64
                elif field == 'source':
                    changed['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
                else:
                    changed['code_commit'] = '0' * 40
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    validate_frozen(self.c, plan, changed)

    def test_matching_current_owner_full_completion_required(self):
        for name, cid in (('server-1', 'AdaRES-Y92'), ('server-2', 'AdaRES-Y93')):
            proof = dict(id=cid, complete=True, process_absent=True, CUDA_initialized=False, completed_epochs={phase: list(range(1, 31)) for phase in ('Train', 'Val')}, checkpoints=[dict(epoch=epoch, scheduler_last_epoch=epoch, optimizer_state_count=1, model_finite=True, optimizer_finite=True) for epoch in (30, 21)])
            validate_terminal(proof, name)
            for key, value in (('process_absent', False), ('CUDA_initialized', True), ('id', 'AdaRES-Y91')):
                with self.subTest(name=name, key=key), self.assertRaises(AssertionError):
                    validate_terminal(dict(proof, **{key: value}), name)
            proof['checkpoints'][0]['scheduler_last_epoch'] = 29
            with self.assertRaises(AssertionError):
                validate_terminal(proof, name)

    def test_actual_retained_current_HEAD_owner_binding(self):
        binding = tick.load(ROOT / self.c['source_manifest']['path'])
        validate_binding(binding, self.c, PARENT)
        for key, value in (('working_HEAD', '0' * 40), ('CPU_tested_scientific_files_exact', False), ('matching_current_owners', {'server-2': 'AdaRES-Y91'}), ('main_index_unchanged', False)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_binding(dict(binding, **{key: value}), self.c, PARENT)

    def test_actual_helper_and_GPU1_pause_bound(self):
        for plan, frozen, _ in self.values.values():
            self.assertEqual(plan['execution_helper'], ref('scripts/run_y98_native.py'))
            self.assertEqual(plan['additive_parent'], PARENT)
        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'], '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')
        self.assertEqual(self.servers['server-6']['allowed_gpus'], [])
if __name__ == '__main__':
    unittest.main()
