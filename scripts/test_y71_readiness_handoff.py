"""Reject occupied/reserved GPU0 without treating the separate Y57 pause as a slot."""
import copy
import unittest
from run_y71_gpu0_probe import slot_blockers,physically_idle
class Readiness(unittest.TestCase):
    def setUp(self):
        self.pause=dict(id='AdaRES-Y57',status='paused',gpu='1',dispatch_hold='user pause',resume_requires_user_authorization=True)
        self.queue=dict(gpu_queues={'server-2':[self.pause,dict(id='AdaRES-Y68',status='running',gpu='0'),dict(id='AdaRES-Y70',status='running',gpu='1')]})
        self.running={'server-2':{'0':'AdaRES-Y68','1':'AdaRES-Y70'}}
        self.live=dict(hostname='dl01-MS-7C60',power_limit_watts=280.,memory_used_mib=98.,utilization_percent=0.,compute_pids=[])
    def test_running_slot_blocked(self):self.assertEqual(slot_blockers(self.queue,self.running),['AdaRES-Y68'])
    def test_claiming_or_waiting_slot_blocked(self):
        for status in ('waiting','claiming'):
            self.queue['gpu_queues']['server-2'][1]['status']=status
            self.assertEqual(slot_blockers(self.queue,{}),['AdaRES-Y68'])
    def test_paused_gpu1_does_not_reserve_gpu0(self):
        self.queue['gpu_queues']['server-2'].pop(1)
        self.assertEqual(slot_blockers(self.queue,{'server-2':{'1':'AdaRES-Y70'}}),[])
        self.assertEqual(self.pause['status'],'paused')
    def test_foreign_mapping_preserved(self):self.assertIn('foreign',slot_blockers(self.queue,{'server-2':{'0':'foreign'}}))
    def test_pause_changes_refused(self):
        for key,value in (('status','running'),('dispatch_hold',''),('resume_requires_user_authorization',False)):
            old=copy.deepcopy(self.pause);self.pause[key]=value
            with self.assertRaises(AssertionError):slot_blockers(self.queue,self.running)
            self.pause.clear();self.pause.update(old)
    def test_idle_schema_only(self):self.assertTrue(physically_idle(self.live))
    def test_compute_memory_or_utilization_blocks(self):
        for key,value in (('compute_pids',[20742]),('memory_used_mib',1024.),('utilization_percent',10.)):
            self.assertFalse(physically_idle(dict(self.live,**{key:value})))
    def test_wrong_host_power_or_negative_measurement_refused(self):
        for key,value in (('hostname','other'),('power_limit_watts',390.),('memory_used_mib',-1.),('utilization_percent',101.)):
            with self.assertRaises(AssertionError):physically_idle(dict(self.live,**{key:value}))
if __name__=='__main__':unittest.main()
