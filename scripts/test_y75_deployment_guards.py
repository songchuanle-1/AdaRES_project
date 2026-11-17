"""Pure deployment/resource guards; fixture success is not a native GPU PASS."""
import copy
import unittest
import y75_deployment as deploy
import advance_y75 as advance

class Guards(unittest.TestCase):
    def test_actual_registered_chain_and_keeper_bytes(self):
        for cid,commit in (('AdaRES-Y72',deploy.Y72),('AdaRES-Y74',deploy.Y74)):
            self.assertIn(deploy.Y75,deploy.contract(commit,deploy.Y75,cid))
    def test_only_exact_seven_added_candidate_files(self):
        rows=[['A',p] for p in deploy.paths('AdaRES-Y75')]
        deploy.validate_delta(rows,'AdaRES-Y75')
        for bad in (rows+[['A','foreign.py']],[['M',rows[0][1]]]+rows[1:],rows[:-1]):
            with self.assertRaises(AssertionError):deploy.validate_delta(bad,'AdaRES-Y75')
    def test_unknown_or_reverse_version_refused(self):
        for a,b in ((deploy.Y72,'0'*40),('0'*40,deploy.Y75),(deploy.Y75,deploy.Y74)):
            with self.assertRaises(AssertionError):deploy.contract(a,b,'AdaRES-Y72')
    def test_paused_Y57_does_not_reserve_GPU1(self):
        paused=dict(id='AdaRES-Y57',gpu=1,status='paused',dispatch_hold=True,resume_requires_user_authorization=True)
        running=dict(id='AdaRES-Y74',gpu=0,status='running')
        advance.unreserved(dict(gpu_queues={'server-2':[paused,running]}),{'server-2':{'0':'AdaRES-Y74'}},'server-2',1)
    def test_foreign_queue_or_mapping_GPU1_refused(self):
        paused=dict(id='AdaRES-Y57',gpu=1,status='paused',dispatch_hold=True,resume_requires_user_authorization=True)
        for status in ('waiting','claiming','running'):
            q=dict(gpu_queues={'server-2':[paused,dict(id='foreign',gpu=1,status=status)]})
            with self.assertRaises(AssertionError):advance.unreserved(q,{},'server-2',1)
        with self.assertRaises(AssertionError):advance.unreserved(dict(gpu_queues={'server-2':[paused]}),{'server-2':{'1':'foreign'}},'server-2',1)
    def test_unknown_occupation_foreign_compute_or_changed_power_refused(self):
        advance.idle(dict(gpu='0,1,390',compute=''),390.)
        for gpu,compute in (('nan,0,390',''),('0,0,280',''),('0,2000,390',''),('0,0,390','123')):
            with self.assertRaises(AssertionError):advance.idle(dict(gpu=gpu,compute=compute),390.)

if __name__=='__main__':unittest.main()
