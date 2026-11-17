"""Resource adaptation controls for Y84; no real GPU task in these fixtures."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import y84_server1_alternative as alt

BODY = Path(__file__).with_name('probe_y84_native_smoke.py').read_text(encoding='utf-8')
MAPPINGS = [dict(source='/old/data', target='/new/data'), dict(source='/old/work', target='/new/work')]


class Alternative(unittest.TestCase):
    def test_body_reversible_exact(self):
        changed = alt.adapted_body(BODY, 350.)
        self.assertEqual(changed.replace(alt.ADAPT, '').replace("== 'dl02-MS-7B73'", "== 'dl01-MS-7C60'"), BODY)

    def test_only_mapped_paths_change(self):
        cfg = SimpleNamespace(train=SimpleNamespace(pretrain='/old/data/backbone.pth', seed=1999, epochs=30),
            work_dir='/old/work', data={s: SimpleNamespace(data_root='/old/data/') for s in ('train', 'val')},
            model={'channels': 256}, optimizer={'lr': .0001})
        ns = dict(P={'path_mappings': MAPPINGS}, FROZEN={'runtime_path_mappings': copy.deepcopy(MAPPINGS)}, cfg=cfg)
        exec(alt.ADAPT, ns)
        self.assertEqual(cfg.train.pretrain, '/new/data/backbone.pth')
        self.assertEqual(cfg.work_dir, '/new/work')
        self.assertEqual([cfg.data[s].data_root for s in ('train', 'val')], ['/new/data/'] * 2)
        self.assertEqual((cfg.train.seed, cfg.train.epochs, cfg.model, cfg.optimizer), (1999, 30, {'channels': 256}, {'lr': .0001}))
        self.assertEqual(ns['adapt_path']('/old/database/file'), '/old/database/file')

    def test_unbound_mapping_rejected(self):
        with self.assertRaises(AssertionError):
            exec(alt.ADAPT, dict(P={'path_mappings': []}, FROZEN={'runtime_path_mappings': MAPPINGS},
                cfg=SimpleNamespace(train=SimpleNamespace(pretrain='/old/data/a'))))

    def test_wrong_power_host_or_multiple_loaders_rejected(self):
        for power in (280., 390., float('nan')):
            with self.assertRaises(AssertionError): alt.adapted_body(BODY, power)
        for body in (BODY.replace('dl01-MS-7C60', 'foreign'),
            BODY.replace("== FROZEN['runtime_power_limit_watts']", '== 280.0'),
            BODY + '\ncfg = gorilla.Config.fromfile(str(config_path))\n'):
            with self.assertRaises(AssertionError): alt.adapted_body(body, 350.)

    def test_only_exact_idle_telemetry_accepted(self):
        obs = dict(hostname='dl02-MS-7B73', gpu='0, 0, 350', compute='')
        self.assertTrue(alt.idle(obs))
        for gpu in ('1024, 0, 350', '0, 10, 350', '17294, 99, 350'):
            self.assertFalse(alt.idle(dict(obs, gpu=gpu)))
        self.assertFalse(alt.idle(dict(obs, compute='1121817\n')))

    def test_foreign_host_power_or_invalid_telemetry_rejected(self):
        obs = dict(hostname='dl02-MS-7B73', gpu='0, 0, 350', compute='')
        for bad in (dict(obs, hostname='foreign'), dict(obs, gpu='0, 0, 280'),
            dict(obs, gpu='nan, 0, 350'), dict(obs, gpu='inf, 0, 350'), dict(obs, gpu='0, -1, 350')):
            with self.assertRaises(AssertionError): alt.idle(bad)

    def test_waiting_claiming_running_and_map_reservations_block(self):
        for status in ('waiting', 'claiming', 'running'):
            queue = {'gpu_queues': {'server-1': [dict(id='AdaRES-Y82', status=status, gpu='0')]}}
            self.assertEqual(alt.blockers(queue, {}), ['AdaRES-Y82'])
            with self.assertRaises(AssertionError):
                alt.require_available(queue, {}, dict(hostname='dl02-MS-7B73', gpu='0, 0, 350', compute=''))
        with self.assertRaises(AssertionError):
            alt.require_available({'gpu_queues': {'server-1': []}}, {'server-1': {'0': 'AdaRES-Y82'}},
                dict(hostname='dl02-MS-7B73', gpu='0, 0, 350', compute=''))

    def test_unknown_device_and_busy_card_never_available(self):
        with self.assertRaises(AssertionError):
            alt.blockers({'gpu_queues': {'server-1': []}}, {'server-1': {'1': 'foreign'}})
        queue = {'gpu_queues': {'server-1': []}}
        with self.assertRaises(AssertionError):
            alt.require_available(queue, {}, dict(hostname='dl02-MS-7B73', gpu='0, 0, 350', compute='1121817'))
        alt.require_available(queue, {}, dict(hostname='dl02-MS-7B73', gpu='0, 0, 350', compute=''))


if __name__ == '__main__':
    unittest.main()
