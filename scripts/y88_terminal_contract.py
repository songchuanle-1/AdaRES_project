"""Exact original AdaRES-Y88 terminal source and checkpoint identity."""
from current_terminal_contract import IDENTITY, Y87, Y88, paths, validate_delta
from current_terminal_contract import build_contract as bound_contract, validate_marker

SPEC = {'server': 'server-1', 'pid': 3527278, 'code_commit': 'cb4a5813755d38b3d1593bb803726aabd7d0a79b', 'package': 'adares_y88_mask_query_quadratic', 'log': '/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y88_mask_query_quadratic_res_20261106_053534.log', 'admission_sha256': '39f9a71debe48314d479b42b604565cf557ccbe40e5582e949a66959587b3572', 'freeze_sha256': '93bf51b65a580cbdd5a7ade69fae370d8d0e6e6db5ab665645892c775d84aa81', 'config': 'configs_song/model_paper5/adares_y88_mask_query_quadratic_res.yaml', 'id': 'AdaRES-Y88', 'checkpoint_directory': '20261106_053553', 'location_evidence': {'path': 'pipeline/checks/y88_terminal_original_location_75d4b121caff4ecab72bfeac9601eefe.json', 'sha256': '8b1fe954b39a55a48a57028f86528458ca5105dddf5d2a63d5f8e2c8ad63136f'}}
CID = SPEC["id"]
CHECKPOINT_DIRECTORY = SPEC["checkpoint_directory"]
CONFIG = SPEC["config"]

def build_contract(task, server):
    return bound_contract(SPEC, task, server)
