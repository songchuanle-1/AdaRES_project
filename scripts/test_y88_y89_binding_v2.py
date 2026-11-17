"""Correct only the expected exception type from the existing resource guard."""
import copy
import unittest
import test_y88_y89_binding as previous


class AdditiveBinding(previous.AdditiveBinding):
    def test_original_runtime_and_mappings_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in [('python', '/wrong/python'), ('runtime_library_path', '/wrong/lib'),
                ('allowed_gpus', [0, 1]), ('runtime_path_mappings', [])]:
            changed = copy.deepcopy(server)
            changed[key] = value
            expected = ValueError if key == 'allowed_gpus' else AssertionError
            with self.subTest(key=key), self.assertRaises(expected):
                previous.binding.build_contract(task, changed)


if __name__ == '__main__':
    unittest.main()
