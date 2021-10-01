import logging

import MinkowskiEngine as ME
import numpy as np
import torch


class CollationFunctionFactory:
    def __init__(self, collation_type="collate_default"):
        if collation_type == "collate_default":
            self.collation_fn = self.collate_default
        elif collation_type == "collate_pair":
            self.collation_fn = self.collate_pair
        else:
            raise ValueError(f"collation_type {collation_type} not found")

    def __call__(self, list_data):
        return self.collation_fn(list_data)

    def collate_default(self, list_data):
        return list_data

    def collate_pair(self, list_data):
        N = len(list_data)
        list_data = [data for data in list_data if data is not None]
        if N != len(list_data):
            logging.info(f"Retain {len(list_data)} from {N} data.")
        if len(list_data) == 0:
            raise ValueError("No data in the batch")

        xyz0, xyz1, C0, C1, F0, F1, trans, extra_packages = list(zip(*list_data))
        trans_batch, len_batch = [], []

        coords_batch0 = ME.utils.batched_coordinates(C0)
        coords_batch1 = ME.utils.batched_coordinates(C1)
        trans_batch = torch.from_numpy(np.stack(trans)).float()

        len_batch = [[c0.shape[0], c1.shape[0]] for c0, c1 in zip(C0, C1)]

        # Concatenate all lists
        feats_batch0 = torch.cat(F0, 0).float()
        feats_batch1 = torch.cat(F1, 0).float()

        return {
            "pcd0": torch.cat(xyz0, 0),
            "pcd1": torch.cat(xyz1, 0),
            "sinput0_C": coords_batch0,
            "sinput0_F": feats_batch0,
            "sinput1_C": coords_batch1,
            "sinput1_F": feats_batch1,
            "T_gt": trans_batch,
            "len_batch": len_batch,
            "extra_packages": extra_packages,
        }
