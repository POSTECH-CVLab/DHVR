import logging

import gin
import MinkowskiEngine as ME
import numpy as np
import torch
from torch_batch_svd import svd as fast_svd

import src.utils.geometry as geometry
from src.utils.knn import feature_matching
from src.utils.misc import random_triplet


@gin.configurable()
class DHVR:
    def __init__(
        self,
        device,
        feature_extractor,
        refine_model,
        voxel_size=0.05,
        num_trial=100000,
        smoothing=False,
        kernel_size=3,
        r_binsize=0.02,
        t_binsize=0.02,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

    def sample_triplet(self, xyz0, xyz1, F0, F1):
        # feature matching in both direction
        pairs = feature_matching(F0, F1, mutual=False)
        pairs_inv = feature_matching(F1, F0, mutual=False)
        pairs = torch.cat([pairs, pairs_inv.roll(1, 1)], dim=0)

        # sample random triplets
        triplets = random_triplet(len(pairs), self.num_trial * 5)

        # check geometric constraints
        idx0 = pairs[triplets, 0]
        idx1 = pairs[triplets, 1]
        xyz0_sel = xyz0[idx0].reshape(-1, 3, 3)
        xyz1_sel = xyz1[idx1].reshape(-1, 3, 3)
        li = torch.norm(xyz0_sel - xyz0_sel.roll(1, 1), p=2, dim=2)
        lj = torch.norm(xyz1_sel - xyz1_sel.roll(1, 1), p=2, dim=2)

        triangle_check = torch.all(
            torch.abs(li - lj) < 3 * self.voxel_size, dim=1
        ).cpu()
        dup_check = torch.logical_and(
            torch.all(li > self.voxel_size * 1.5, dim=1),
            torch.all(lj > self.voxel_size * 1.5, dim=1),
        ).cpu()

        triplets = triplets[torch.logical_and(triangle_check, dup_check)]

        if triplets.shape[0] > self.num_trial:
            idx = np.round(
                np.linspace(0, triplets.shape[0] - 1, self.num_trial)
            ).astype(int)
            triplets = triplets[idx]
        return pairs, triplets

    def solve(self, xyz0, xyz1, pairs, triplets):
        xyz0_sel = xyz0[pairs[triplets, 0]]
        xyz1_sel = xyz1[pairs[triplets, 1]]

        # zero mean shift
        xyz0_mean = xyz0_sel.mean(1, keepdim=True)
        xyz1_mean = xyz1_sel.mean(1, keepdim=True)
        xyz0_centered = xyz0_sel - xyz0_mean
        xyz1_centered = xyz1_sel - xyz1_mean

        # solve rotation
        H = xyz1_centered.transpose(1, 2) @ xyz0_centered
        U, D, V = fast_svd(H)
        S = torch.eye(3).repeat(U.shape[0], 1, 1).to(U.device)
        det = U.det() * V.det()
        S[det < 0, -1, -1] = -1
        Rs = U @ S @ V.transpose(1, 2)
        angles = geometry.rotation_to_axis_angle(Rs)

        # solve translation using centroid
        xyz0_rotated = torch.bmm(Rs, xyz0_mean.permute(0, 2, 1)).squeeze(2)
        t = xyz1_mean.squeeze(1) - xyz0_rotated

        return angles, t

    def vote(self, Rs, ts):
        r_coord = torch.floor(Rs / self.r_binsize)
        t_coord = torch.floor(ts / self.t_binsize)
        coord = torch.cat(
            [
                torch.zeros(r_coord.shape[0]).unsqueeze(1).to(self.device),
                r_coord,
                t_coord,
            ],
            dim=1,
        )
        feat = torch.ones(coord.shape[0]).unsqueeze(1).to(self.device)
        vote = ME.SparseTensor(
            feat.float(),
            coordinates=coord.int(),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM,
        )
        return vote

    def evaluate(self, vote):
        max_index = vote.F.squeeze(1).argmax()
        max_value = vote.C[max_index, 1:]
        angle = (max_value[:3] + 0.5) * self.r_binsize
        t = (max_value[3:] + 0.5) * self.t_binsize
        R = geometry.axis_angle_to_rotation(angle)
        return R, t

    def register(self, pcd0, pcd1, coord0=None, coord1=None, feat0=None, feat1=None):
        F0, xyz0 = self.feature_extractor.extract_feature(pcd0, coord0, feat0)
        F1, xyz1 = self.feature_extractor.extract_feature(pcd1, coord1, feat1)

        # sample correspodences
        pairs, combinations = self.sample_triplet(xyz0, xyz1, F0, F1)

        try:
            # solve R, t
            angles, ts = self.solve(xyz0, xyz1, pairs, combinations)

            # rotation & translation voting
            votes = self.vote(angles, ts)

            # gaussian smoothing
            if self.smoothing:
                votes = geometry.sparse_gaussian(
                    votes, kernel_size=self.kernel_size, dimension=6
                )

            # post processing
            if self.refine_model is not None:
                votes = self.refine_model(votes)
            R, t = self.evaluate(votes)
            self.hspace = votes
            T = torch.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t

            # empty cache
            torch.cuda.empty_cache()
        except Exception as e:
            logging.exception(e)
            import pdb

            pdb.set_trace()
            return torch.eye(4)

        return T
