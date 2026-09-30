"""Local training dataset integration; native validation remains unchanged."""

from ipdn.dataset import ScanNetDataset_sample_graph_edge
from .pairing import encoded_pairs, paired_chunks


class PairedTrainingDataset(ScanNetDataset_sample_graph_edge):
    def __init__(self, **kwargs):
        if not kwargs.get("training") or kwargs.get("prefix") != "train":
            raise ValueError("Y17 pairing is training-only")
        super().__init__(**kwargs)
        self.scene_inputs = paired_chunks(self.scene_inputs)

    def collate_fn(self, batch):
        result = super().collate_fn(batch)
        result["target_pairs"] = encoded_pairs(
            result["object_idss"], result["lang_tokenss"], result["scenes_len"])
        return result


def build_train_dataset(data_cfg, logger):
    kwargs = data_cfg.copy()
    if kwargs.pop("type") != "scannetv2_sample_graph_edge":
        raise ValueError("Y17 requires the frozen native dataset")
    return PairedTrainingDataset(logger=logger, **kwargs)
