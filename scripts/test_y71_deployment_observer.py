"""Keep exact identities and markers strict after the additive Y71 deployment."""
import copy
import unittest
from observe_y70_y71_exact_deployment import validate_task, validate_marker, OLD, NEW


class Contracts(unittest.TestCase):
    def task(self, cid):
        return dict(id=cid, status='running', launch_verified=True,
            gpu=1 if cid == 'AdaRES-Y70' else 0, pid=52561 if cid == 'AdaRES-Y70' else 23456,
            code_commit=OLD if cid == 'AdaRES-Y70' else NEW,
            config='configs_song/model_paper5/' + ('adares_y70_query_scale_fusion' if cid == 'AdaRES-Y70' else 'adares_y71_visual_product_fusion') + '_res.yaml')

    def test_exact_schema(self):
        self.assertEqual(validate_task(self.task('AdaRES-Y70'))[1:], (1, 390., OLD))
        self.assertEqual(validate_task(self.task('AdaRES-Y71'))[1:], (0, 280., NEW))
        for marker in (OLD, NEW): validate_marker(marker, 'AdaRES-Y70')
        validate_marker(NEW, 'AdaRES-Y71')

    def test_wrong_or_uncertain_state(self):
        for cid in ('AdaRES-Y70', 'AdaRES-Y71'):
            for key, value in (('status', 'claiming'), ('launch_verified', False), ('launch_verified', 1)):
                t = self.task(cid); t[key] = value
                with self.assertRaises(AssertionError): validate_task(t)

    def test_wrong_gpu_config_commit(self):
        for cid in ('AdaRES-Y70', 'AdaRES-Y71'):
            for key, value in (('gpu', 2), ('config', 'other.yaml'), ('code_commit', 'f' * 40)):
                t = self.task(cid); t[key] = value
                with self.assertRaises(AssertionError): validate_task(t)

    def test_changed_keeper_or_invalid_new_pid(self):
        for cid, values in (('AdaRES-Y70', (52562, 52561.)), ('AdaRES-Y71', (0, -1, True, 23456.))):
            for value in values:
                t = self.task(cid); t['pid'] = value
                with self.assertRaises(AssertionError): validate_task(t)

    def test_arbitrary_descendant_or_old_new_task_marker(self):
        for cid in ('AdaRES-Y70', 'AdaRES-Y71'):
            for marker in ('f' * 40, 'c60c903941d6cd7ffcc343d7c13610f2ba416caa'):
                with self.assertRaises(AssertionError): validate_marker(marker, cid)
        with self.assertRaises(AssertionError): validate_marker(OLD, 'AdaRES-Y71')

    def test_foreign_task(self):
        for cid in ('AdaRES-Y68', 'AdaQ3D-Y71'):
            t = self.task('AdaRES-Y71'); t['id'] = cid
            with self.assertRaises(AssertionError): validate_task(t)
            with self.assertRaises(AssertionError): validate_marker(NEW, cid)


if __name__ == '__main__': unittest.main()
