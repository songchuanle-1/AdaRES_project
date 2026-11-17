"""Exact original AdaRES-Y87 terminal source and checkpoint identity."""
from current_terminal_contract import IDENTITY, Y87, Y88, paths, validate_delta
from current_terminal_contract import build_contract as bound_contract, validate_marker

SPEC = {'server': 'server-2', 'pid': 26281, 'code_commit': '47ec0813cb4131ddf3e0a08db4fa9ac877123685', 'package': 'adares_y87_self_language_product', 'log': '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y87_self_language_product_res_20261106_032555.log', 'admission_sha256': '300d46379e6072066f775c03ceaf2142d806a9fe02c3ad88f742126f5263e30c', 'freeze_sha256': 'b5aab97e1d694e64315cf0a0687a2c8c35a6eb5e73f5915cbffb9f36b9eaa3e4', 'config': 'configs_song/model_paper5/adares_y87_self_language_product_res.yaml', 'id': 'AdaRES-Y87', 'checkpoint_directory': '20261105_142459', 'location_evidence': {'path': 'pipeline/checks/y87_terminal_original_location_75d4b121caff4ecab72bfeac9601eefe.json', 'sha256': 'bb64984ce75cf93e79dcbb1368337c5f4715e3c8e9c6a58f8f56545331dbdf3e'}}
CID = SPEC["id"]
CHECKPOINT_DIRECTORY = SPEC["checkpoint_directory"]
CONFIG = SPEC["config"]

def build_contract(task, server):
    return bound_contract(SPEC, task, server)
