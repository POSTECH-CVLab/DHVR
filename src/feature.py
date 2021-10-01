import logging
import os
from abc import ABC

import gin
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch

from src.models import get_model


class BaseFeatureExtractor(ABC):
    def __init__(self):
        logging.info(f"Initialize {self.__class__.__name__}")

    def extract_feature(self, xyz):
        raise NotImplementedError("Feature should implement extract_feature method.")


@gin.configurable()
class FCGF(BaseFeatureExtractor):
    def __init__(self, voxel_size, checkpoint_path, device):
        super().__init__()
        self.voxel_size = voxel_size
        self.device = device
        assert os.path.exists(checkpoint_path), f"{checkpoint_path} not exists"

        MODEL = get_model("ResUNetBN2C")
        feat_model = MODEL(
            1, 32, bn_momentum=0.05, conv1_kernel_size=7, normalize_feature=True
        ).to(device)
        checkpoint = torch.load(checkpoint_path)
        feat_model.load_state_dict(checkpoint["state_dict"])
        self.feat_model = feat_model
        self.feat_model.eval()

    def freeze(self):
        for param in self.feat_model.parameters():
            param.requires_grad = False

    def extract_feature(self, xyz, coords=None, feats=None):
        if coords is None or feats is None:
            # quantize input xyz.
            coords, sel = ME.utils.sparse_quantize(
                xyz / self.voxel_size, return_index=True
            )

            # make sparse tensor.
            coords = ME.utils.batched_coordinates([coords])
            feats = torch.ones((coords.shape[0], 1)).float()
            sinput = ME.SparseTensor(
                feats.to(self.device), coordinates=coords.to(self.device)
            )
            if isinstance(xyz, np.ndarray):
                xyz = torch.from_numpy(xyz)
            xyz = xyz[sel].float().to(self.device)
        else:
            sinput = ME.SparseTensor(coordinates=coords, features=feats)

        # extract feature.
        F = self.feat_model(sinput).F

        return F, xyz


@gin.configurable()
class FPFH(BaseFeatureExtractor):
    def __init__(self, voxel_size, device):
        super().__init__(voxel_size, device)

    def extract_feature(self, xyz):
        voxel_size = self.voxel_size
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.numpy()

        # downsample
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd = pcd.voxel_down_sample(voxel_size)

        # calculate normals
        radius_normal = voxel_size * 2.0
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

        # calculate features
        radius_feature = voxel_size * 5.0
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        xyz = torch.from_numpy(np.asarray(pcd.points)).float()
        F = torch.from_numpy(pcd_fpfh.data.copy().T).float().contiguous()
        return F, xyz


MODELS = [FPFH, FCGF]


@gin.configurable()
def get_feature(name):
    # Find the model class from its name
    all_models = MODELS
    mdict = {model.__name__: model for model in all_models}
    if name not in mdict:
        logging.info(f"Invalid model index. You put {name}. Options are:")
        # Display a list of valid model names
        for model in all_models:
            logging.info("\t* {}".format(model.__name__))
        return None
    NetClass = mdict[name]

    return NetClass
