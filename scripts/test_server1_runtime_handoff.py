"""Negative controls for the new, explicitly authorized server deployment."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import closure
import source_integrity


class ServerGrantTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'AdaRES_project'
        self.root.mkdir()
        self.grant=dict(project='AdaRES_project',authorization_granted=True,server='server-1',
            host='192.168.1.175',allowed_gpus=[0],expected_hostname='dl02-MS-7B73')
        (self.root/'grant.json').write_text(json.dumps(self.grant))
        self.server=dict(name='server-1',host='192.168.1.175',allowed_gpus=[0],
            expected_hostname='dl02-MS-7B73',max_tasks_per_gpu=1,resource_authorization='grant.json')

    def test_exact_explicit_grant_is_accepted(self):
        closure.validate_adares_server(self.root,self.server)

    def test_wrong_host_or_gpu_or_hostname_is_rejected(self):
        for key,value in [('host','192.168.1.163'),('allowed_gpus',[0,1]),
                          ('expected_hostname','other'),('max_tasks_per_gpu',2),
                          ('resource_authorization',None)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                closure.validate_adares_server(self.root,dict(self.server,**{key:value}))

    def test_server6_and_other_projects_stay_rejected(self):
        with self.assertRaises(ValueError):
            closure.validate_adares_server(self.root,dict(self.server,name='server-6'))
        with self.assertRaises(ValueError):
            closure.validate_adares_server(self.root.parent/'other',self.server)


class MappedSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.code=self.root/'AdaRES'
        self.code.mkdir()
        self.git('init','-q')
        (self.code/'model.py').write_text('weight=1\n')
        (self.code/'config.yaml').write_text('root: /old/data/\nepochs: 30\n')
        self.git('add','.')
        self.git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')
        self.commit=self.git('rev-parse','HEAD')
        (self.code/'.codex_source_commit').write_text(self.commit)
        self.server=dict(host='unused',user='unused',remote_code_root=str(self.code),
            runtime_path_mappings=[dict(source='/old/data',target='/new/data')])

    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.root),*args],stderr=subprocess.DEVNULL).decode().strip()

    def remote(self,host,user,command):
        import shlex
        program=shlex.split(command)[-1]
        return subprocess.run([sys.executable,'-B','-c',program],capture_output=True,text=True).stdout

    def test_original_and_exact_path_adaptation_pass(self):
        source_integrity.verify_marker(self.root,self.commit,self.server,self.remote)
        (self.code/'config.yaml').write_text('root: /new/data/\nepochs: 30\n')
        source_integrity.verify_marker(self.root,self.commit,self.server,self.remote)

    def test_changed_schedule_is_rejected_after_path_adaptation(self):
        (self.code/'config.yaml').write_text('root: /new/data/\nepochs: 31\n')
        with self.assertRaises(ValueError):
            source_integrity.verify_marker(self.root,self.commit,self.server,self.remote)

    def test_changed_model_is_rejected(self):
        (self.code/'model.py').write_text('weight=2\n')
        with self.assertRaises(ValueError):
            source_integrity.verify_marker(self.root,self.commit,self.server,self.remote)


if __name__=='__main__':
    unittest.main()
