"""Guards against removing the successor when closing its predecessor."""
import copy
import unittest
from reconcile_y62_y63_server1 import validate_mapping


class SuccessorOwnership(unittest.TestCase):
    def test_preserves_successor_and_other_servers_without_aliasing(self):
        current = {'server-1': {'0': 'AdaRES-Y63'}, 'server-2': {}}
        result = validate_mapping(current)
        self.assertEqual(result, current)
        result['server-1'].clear()
        self.assertEqual(current['server-1']['0'], 'AdaRES-Y63')

    def test_rejects_missing_or_changed_successor(self):
        for slot in ({}, {'0': 'AdaRES-Y62'}, {'0': 'foreign'}, {'1': 'AdaRES-Y63'}):
            with self.subTest(slot=slot), self.assertRaises(AssertionError):
                validate_mapping({'server-1': slot})

    def test_rejects_old_experiment_in_another_slot(self):
        for server, gpu in [('server-1', '1'), ('server-2', '0')]:
            current = {'server-1': {'0': 'AdaRES-Y63'}, 'server-2': {}}
            current[server][gpu] = 'AdaRES-Y62'
            before = copy.deepcopy(current)
            with self.assertRaises(AssertionError): validate_mapping(current)
            self.assertEqual(current, before)


if __name__ == '__main__': unittest.main()
