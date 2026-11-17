"""Exercise admission guards against actual Y98 native observations."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, monitor, ref, tick
from check_ef_execution import validate_frozen
import advance_y98_v2 as advance
import y98_deployment as deploy

class Admission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=tick.load(ROOT/"pipeline/research_work.json")
        cls.c,cls.plan,_,cls.f=advance.prepared(cls.work)
        assert cls.c.get("native_smoke_result")
        cls.raw=tick.load(ROOT/cls.c["native_smoke_result"]["path"])
        parent=next(row for row in cls.work["candidates"] if row["id"]=="AdaRES-Y95")
        cls.parentref=parent["post_training_freeze"]
        cls.parent=tick.load(ROOT/cls.parentref["path"])

    def check(self,value=None):
        advance.validate_result(self.raw if value is None else value,self.c,self.f)

    def test_actual_native_runtime_and_seven_files(self):
        self.check()
        self.assertEqual((self.f["runtime_server"],self.f["runtime_power_limit_watts"]),("server-2",280.))
        self.assertEqual(validate_frozen(self.c,self.plan,self.f),7)
        self.assertEqual(deploy.git("rev-parse",deploy.Y98+"^").decode().strip(),deploy.Y96)
        rows=[["A",path] for path in deploy.paths("AdaRES-Y98")]
        deploy.validate_delta(rows,"AdaRES-Y98")
        for bad in (rows[:-1], rows+[["A","foreign.py"]], [["M",rows[0][1]]]+rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad,"AdaRES-Y98")
        self.assertEqual(list(deploy.contract(deploy.Y96,deploy.Y98,"AdaRES-Y96")),[deploy.Y96,deploy.Y98])

    def test_identity_version_goal_yaml(self):
        for key,value in (("id","foreign"),("code_commit","0"*40),("config","wrong.yaml"),
                          ("goal_sha256","0"*64),("native_YAML_sha256","0"*64)):
            with self.subTest(key=key),self.assertRaises(AssertionError):
                self.check(dict(self.raw,**{key:value}))

    def test_two_real_train_steps_finite_losses_and_no_validation(self):
        for key,value in (("split","val"),("native_entry","synthetic"),("steps",1),
                          ("optimizer_steps",1),("losses",[1.]),("losses",[1.,float("nan")]),
                          ("validation_used",True),("model_weights_saved",True),("admission",True)):
            with self.subTest(key=key),self.assertRaises(AssertionError):
                self.check(dict(self.raw,**{key:value}))

    def test_observed_original_and_new_paths(self):
        for key in ("complete","native_data_loaded","backward_finite","intended_path_observed",
                    "original_base_mask_and_static_reuse_preserved","original_hard_gate_algorithm_preserved","native_modes_verified"):
            with self.subTest(key=key),self.assertRaises(AssertionError):
                self.check(dict(self.raw,**{key:False}))
        for key in ("appearance_mask_calls","original_appearance_projection_calls","original_base_mask_calls",
                    "coarse_gate_consumers","shared_head_calls","query_norm_consumption_calls"):
            with self.subTest(key=key),self.assertRaises(AssertionError):
                self.check(dict(self.raw,**{key:0}))

    def test_per_step_original_gradients_and_updates(self):
        for key,value in (("all_finite",False),("base_adapted_and_original_appearance_gradients_finite",False),
                          ("original_visual_parameter_tensors",0),("appearance_mask_calls",0),
                          ("original_appearance_projection_calls",0),("original_base_mask_calls",0),
                          ("raw_appearance_requires_grad",True),("frozen_text_encoder_preserved",False),
                          ("new_gradient_max",0.),("parameter_update_max",float("nan"))):
            changed=copy.deepcopy(self.raw)
            changed["gradient_checks"][1][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):
                self.check(changed)

    def test_full1024_appearance_and_seven_static_consumers(self):
        for key,value in (("original_full1024_appearance_identity",False),("original_base_mask_identity",False),
                          ("original_appearance_projection_executed",False),("both_inputs_undetached",False),
                          ("float64_full_appearance_projection_reference",False),("original_batching_consumes_skip",False),
                          ("all_seven_predictions_reuse_same_features",False),("weight_nonzero",False),
                          ("raw_appearance_requires_grad",True),("mask_feature_delta_max",0.)):
            changed=copy.deepcopy(self.raw)
            changed["appearance_mask_paths"][1][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):
                self.check(changed)

    def test_exact_data_distinct_scenes_and_device_budget(self):
        changed=copy.deepcopy(self.raw)
        changed["data_source"]["config_data_root"]="/wrong/data"
        with self.assertRaises(AssertionError):
            self.check(changed)
        changed=copy.deepcopy(self.raw)
        changed["batches"][1]=copy.deepcopy(changed["batches"][0])
        with self.assertRaises(AssertionError):
            self.check(changed)
        for value in (0,24*2**30,float("nan")):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw,peak_reserved_bytes=value))

    def test_full_original_server2_protocol_exact(self):
        before=copy.deepcopy(self.parent)
        frozen=advance.make_freeze(self.parent,self.parentref)
        self.assertEqual(self.parent,before)
        self.assertEqual(frozen["training"],self.parent["training"])
        restored=copy.deepcopy(frozen["evaluation"])
        restored["methods"][0]="AdaRES-Y95"
        self.assertEqual(restored,self.parent["evaluation"])
        self.assertEqual(frozen["resource_condition"],self.parent["resource_condition"])
        self.assertEqual(frozen["resource_authorization"],ref(advance.GRANT))
        for key,value in (("id","wrong"),("code_commit","0"*40),("complete",False)):
            with self.assertRaises(AssertionError):
                advance.make_freeze(dict(self.parent,**{key:value}),self.parentref)

    def test_pauses_private_runtime_busy_and_reserved_resources(self):
        servers=monitor.load_servers(str(ROOT))
        self.assertEqual(servers["server-1"]["allowed_gpus"],[0])
        self.assertEqual(servers["server-2"]["allowed_gpus"],[0])
        self.assertEqual(servers["server-6"]["allowed_gpus"],[])
        self.assertEqual(ref(advance.PAUSE)["sha256"],advance.PAUSE_SHA)
        for key,value in (("python","/wrong/python"),("expected_hostname","foreign"),("allowed_gpus",[0,1])):
            with self.assertRaises(AssertionError):
                advance.runtime("server-2",self.c,self.f,dict(servers["server-2"],**{key:value}))
        advance.idle(dict(gpu="0,1,280",compute=""),280.)
        for gpu,compute in (("nan,0,280",""),("0,0,350",""),("0,2000,280",""),("0,0,280","99")):
            with self.assertRaises(AssertionError):
                advance.idle(dict(gpu=gpu,compute=compute),280.)
        queue,running=monitor.load_queue(str(ROOT)),monitor.load_running(str(ROOT))
        advance.unreserved(queue,running,"server-2",0)
        with self.assertRaises(AssertionError):
            advance.unreserved(queue,running,"server-1",0)

    def test_original_complete_checkpoint_and_actual_additive_handoff(self):
        advance.require_completed_owner()
        deploy.require_running_parent_handoff(self.work,self.c)
        changed=copy.deepcopy(self.c)
        changed["native_smoke_result"]=dict(path="missing.json",sha256="0"*64)
        with self.assertRaises(AssertionError):
            deploy.require_running_parent_handoff(self.work,changed)

    def test_python38_helpers(self):
        for name in ("advance_y98_v2.py","y98_deployment.py","y96_y98_terminal_contract_v2.py",
                     "prepare_y96_y98_terminal_binding_v2.py","test_y98_admission.py"):
            ast.parse((ROOT/"scripts"/name).read_text(encoding="utf-8"),feature_version=(3,8))
