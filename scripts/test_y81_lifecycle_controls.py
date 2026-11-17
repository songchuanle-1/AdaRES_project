"""Pre-native GPU1 safety controls; these do not constitute native TRAIN PASS."""
import copy
import unittest
from unittest.mock import patch
import advance_y81 as advance
import y81_deployment as deploy


class Lifecycle(unittest.TestCase):
    def queue(self):
        return dict(gpu_queues={'server-2':[dict(id='AdaRES-Y57',status='paused',gpu=1,
            dispatch_hold=True,resume_requires_user_authorization=True)],'server-1':[]})

    def test_original_gpu1_slot_only(self):
        self.assertEqual(advance.SPECS['AdaRES-Y81'][:3],('server-2',1,390.))

    def test_unknown_occupied_or_changed_power_refused(self):
        advance.idle(dict(gpu='0,1,390',compute=''),390.)
        for gpu,compute in [('nan,1,390',''),('0,1,280',''),('0,1,350',''),('90,22000,390',''),('0,1,390','999')]:
            with self.assertRaises(AssertionError):advance.idle(dict(gpu=gpu,compute=compute),390.)

    def test_busy_reservation_and_queued_task_preserved(self):
        for status in ('waiting','claiming','running'):
            queue=self.queue();queue['gpu_queues']['server-2'].append(dict(id='other',gpu=1,status=status))
            original=copy.deepcopy(queue)
            with self.assertRaises(AssertionError):advance.unreserved(queue,{},'server-2',1)
            self.assertEqual(queue,original)
        with self.assertRaises(AssertionError):advance.unreserved(self.queue(),{'server-2':{'1':'other'}},'server-2',1)

    def test_independent_gpu0_is_preserved(self):
        queue=self.queue();queue['gpu_queues']['server-2'].append(dict(id='AdaRES-Y80',gpu=0,status='running'))
        before=copy.deepcopy(queue)
        advance.unreserved(queue,{'server-2':{'0':'AdaRES-Y80'}},'server-2',1)
        self.assertEqual(queue,before)

    def test_y57_experiment_pause_required(self):
        queue=self.queue();queue['gpu_queues']['server-2'][0]['resume_requires_user_authorization']=False
        with self.assertRaises(AssertionError):advance.unreserved(queue,{},'server-2',1)

    def test_exact_additive_chain_preserves_original_models(self):
        for cid,commit in [('AdaRES-Y79',deploy.Y79),('AdaRES-Y80',deploy.Y80)]:
            self.assertIn(deploy.Y81,deploy.contract(commit,deploy.Y81,cid))
        rows=[['A',p] for p in deploy.paths('AdaRES-Y81')];deploy.validate_delta(rows,'AdaRES-Y81')
        for bad in (rows+[['A','foreign.py']],[['M',rows[0][1]]]+rows[1:]):
            with self.assertRaises(AssertionError):deploy.validate_delta(bad,'AdaRES-Y81')

    def test_unknown_or_reversed_chain_refused(self):
        for commit,head in [(deploy.Y81,deploy.Y80),(deploy.Y79,'0'*40),('0'*40,deploy.Y81)]:
            with self.assertRaises(AssertionError):deploy.contract(commit,head,'AdaRES-Y79')

if __name__=='__main__':unittest.main()
