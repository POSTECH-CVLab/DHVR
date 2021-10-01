import logging

import numpy as np
import torch
import torch.utils.data


class PairDataset(torch.utils.data.Dataset):
    AUGMENT = None

    def __init__(
        self,
        root,
        phase,
        voxel_size=0.05,
        transform=None,
        rotation_range=360,
        random_rotation=True,
        seed=0,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        self.randg = np.random.RandomState()
        if seed is not None:
            self.reset_seed(seed)
        self.files = []

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        t = trans[:3, 3]
        pts = pts @ R.T + t
        return pts

    def __len__(self):
        return len(self.files)
