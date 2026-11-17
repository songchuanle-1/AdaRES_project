"""Resource-only adaptation and refusal to activate over existing training."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import prepare_y69_server1_alternative as alt

BODY = Path(__file__).with_name('probe_y69_native_smoke.py').read_text(encoding='utf-8')
MAPPINGS = [dict(source='/old/data', target='/new/data'), dict(source='/old/work', target='/new/work')]


class AlternativeGuards(unittest.TestCase):
    def test_only_host_power_and_path_changes(self):
        changed = alt.adapted_body(BODY, 350.)
        restored = changed.replace(alt.ADAPT, '').replace("== 'dl02-MS-7B73'", "== 'dl01-MS-7C60'")
        self.assertEqual(restored, BODY)

    def test_exact_mapped_fields_and_original_protocol(self):
        cfg = SimpleNamespace(train=SimpleNamespace(pretrain='/old/data/backbone.pth', seed=1999, epochs=30),
            work_dir='/old/work', data={s: SimpleNamespace(data_root='/old/data/') for s in ('train', 'val')},
            model={'dimension': 256}, optimizer={'lr': .0001})
        ns = dict(P={'path_mappings': MAPPINGS}, FROZEN={'runtime_path_mappings': copy.deepcopy(MAPPINGS)}, cfg=cfg)
        exec(alt.ADAPT, ns)
        self.assertEqual(cfg.train.pretrain, '/new/data/backbone.pth')
        self.assertEqual(cfg.work_dir, '/new/work')
        self.assertEqual([cfg.data[s].data_root for s in ('train', 'val')], ['/new/data/'] * 2)
        self.assertEqual((cfg.train.seed, cfg.train.epochs, cfg.model, cfg.optimizer),
                         (1999, 30, {'dimension': 256}, {'lr': .0001}))
        self.assertEqual(ns['adapt_path']('/old/database/file'), '/old/database/file')

    def test_unbound_path_mapping_refused(self):
        with self.assertRaises(AssertionError):
            exec(alt.ADAPT, dict(P={'path_mappings': []}, FROZEN={'runtime_path_mappings': MAPPINGS},
                 cfg=SimpleNamespace(train=SimpleNamespace(pretrain='/old/data/backbone'))))

    def test_wrong_power_or_unknown_original_guard_refused(self):
        for power in (280., 390., float('nan')):
            with self.subTest(power=power), self.assertRaises(AssertionError): alt.adapted_body(BODY, power)
        for changed in (BODY.replace('dl01-MS-7C60', 'foreign'), BODY.replace("== FROZEN['runtime_power_limit_watts']", '== 280.0'),
                        BODY + '\ncfg = gorilla.Config.fromfile(str(config_path))\n'):
            with self.assertRaises(AssertionError): alt.adapted_body(changed, 350.)

    def test_idle_accepts_only_low_use_and_memory(self):
        obs = dict(hostname='dl02-MS-7B73', gpu='0, 0, 350.00', compute='')
        self.assertTrue(alt.idle(obs))
        for gpu in ('1024, 0, 350', '0, 10, 350', '22756, 100, 350'):
            self.assertFalse(alt.idle(dict(obs, gpu=gpu)))

    def test_compute_owner_blocks_apparently_idle_card(self):
        self.assertFalse(alt.idle(dict(hostname='dl02-MS-7B73', gpu='0, 0, 350', compute='3071037\n')))

    def test_unknown_host_power_or_telemetry_refused(self):
        obs = dict(hostname='dl02-MS-7B73', gpu='0, 0, 350', compute='')
        for changed in (dict(obs, hostname='foreign'), dict(obs, gpu='0, 0, 280'),
                        dict(obs, gpu='nan, 0, 350'), dict(obs, gpu='inf, 0, 350'), dict(obs, gpu='0, -1, 350')):
            with self.assertRaises(AssertionError): alt.idle(changed)

    def test_registered_Y66_prevents_activation_before_remote_or_writes(self):
        frozen = dict(runtime_server='server-1', runtime_power_limit_watts=350., runtime_path_mappings=[],
                      resource_authorization={'path': 'grant', 'sha256': 'unused'})
        program = Mock()
        program.read_text.return_value = 'FROZEN=' + repr(frozen) + '\n'
        candidate = dict(id='AdaRES-Y69', code_commit='9acd73cdc1b96be003c8450a9e22057248a0f531', native_smoke_plan_server1={'path': 'alternate'})
        with patch.object(alt.closure, 'artifact', return_value=program), \
             patch.object(alt.closure, 'validate_adares_server'), \
             patch.object(alt, 'validate_frozen', return_value=7), \
             patch.object(alt.tick, 'load', return_value={'program': {'path': 'frozen'}}), \
             patch.object(alt.monitor, 'load_queue', return_value={'gpu_queues': {'server-1': [
                 {'id': 'AdaRES-Y66', 'status': 'running', 'gpu': '0'}]}}), \
             patch.object(alt, 'observe') as observe, patch.object(alt.tick, 'save') as save:
            with self.assertRaises(AssertionError): alt.activate('unused', {'runtime_path_mappings': []}, {}, candidate)
            observe.assert_not_called()
            save.assert_not_called()


if __name__ == '__main__': unittest.main()
