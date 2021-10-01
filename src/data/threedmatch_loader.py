import glob
import os

import gin
import MinkowskiEngine as ME
import open3d as o3d
from src.data.base_loader import *
from src.data.transforms import *
from src.utils.file import read_trajectory


@gin.configurable()
class ThreeDMatchPairDatasetBase(PairDataset):
    OVERLAP_RATIO = None
    DATA_FILES = None

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
        PairDataset.__init__(
            self,
            root,
            phase,
            voxel_size,
            transform,
            rotation_range,
            random_rotation,
            seed,
        )
        logging.info(f"Loading the subset {phase} from {root}")

        subset_names = open(self.DATA_FILES[phase]).read().split()
        for name in subset_names:
            fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
            fnames_txt = glob.glob(root + "/" + fname)
            assert (
                len(fnames_txt) > 0
            ), f"Make sure that the path {root} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files.append([fname[0], fname[1]])
        logging.info(f"Loaded {len(self.files)} pairs")

    def __getitem__(self, idx):
        file0 = os.path.join(self.root, self.files[idx][0])
        file1 = os.path.join(self.root, self.files[idx][1])
        data0 = np.load(file0)
        data1 = np.load(file1)
        xyz0 = data0["pcd"]
        xyz1 = data1["pcd"]

        if self.random_rotation:
            T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
            T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
            trans = T1 @ np.linalg.inv(T0)

            xyz0 = self.apply_transform(xyz0, T0)
            xyz1 = self.apply_transform(xyz1, T1)
        else:
            trans = np.identity(4)

        # Voxelization
        xyz0_th = torch.from_numpy(xyz0)
        xyz1_th = torch.from_numpy(xyz1)

        _, sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
        _, sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

        # Get features
        npts0 = len(sel0)
        npts1 = len(sel1)

        feats_train0, feats_train1 = [], []

        xyz0_th = xyz0_th[sel0]
        xyz1_th = xyz1_th[sel1]

        feats_train0.append(torch.ones((npts0, 1)))
        feats_train1.append(torch.ones((npts1, 1)))

        F0 = torch.cat(feats_train0, 1)
        F1 = torch.cat(feats_train1, 1)

        C0 = torch.floor(xyz0_th / self.voxel_size)
        C1 = torch.floor(xyz1_th / self.voxel_size)

        if self.transform:
            C0, F0 = self.transform(C0, F0)
            C1, F1 = self.transform(C1, F1)

        extra_package = {"idx": idx, "file0": file0, "file1": file1}

        return (
            xyz0_th.float(),
            xyz1_th.float(),
            C0.int(),
            C1.int(),
            F0.float(),
            F1.float(),
            trans,
            extra_package,
        )


@gin.configurable()
class ThreeDMatchPairDataset03(ThreeDMatchPairDatasetBase):
    OVERLAP_RATIO = 0.3
    DATA_FILES = {
        "train": "./datasets/splits/train_3dmatch.txt",
        "val": "./datasets/splits/val_3dmatch.txt",
        "test": "./datasets/splits/test_3dmatch.txt",
    }


@gin.configurable()
class ThreeDMatchPairDataset05(ThreeDMatchPairDataset03):
    OVERLAP_RATIO = 0.5


@gin.configurable()
class ThreeDMatchPairDataset07(ThreeDMatchPairDataset03):
    OVERLAP_RATIO = 0.7


@gin.configurable()
class ThreeDMatchTestDataset:
    """3DMatch test dataset"""

    DATA_FILES = {"test": "./datasets/splits/test_3dmatch.txt"}
    CONFIG_ROOT = "./datasets/config/3DMatch"
    TE_THRESH = 30
    RE_THRESH = 15

    def __init__(self, root):
        self.root = root

        subset_names = open(self.DATA_FILES["test"]).read().split()
        self.subset_names = subset_names

        self.files = []
        for sname in subset_names:
            traj_file = os.path.join(self.CONFIG_ROOT, sname, "gt.log")
            assert os.path.exists(traj_file)
            traj = read_trajectory(traj_file)
            for ctraj in traj:
                i = ctraj.metadata[0]
                j = ctraj.metadata[1]
                T_gt = ctraj.pose
                self.files.append((sname, i, j, T_gt))
        logging.info(f"Loaded {self.__class__.__name__} with {len(self.files)} data")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sname, i, j, T_gt = self.files[idx]
        file0 = os.path.join(self.root, sname, f"cloud_bin_{i}.ply")
        file1 = os.path.join(self.root, sname, f"cloud_bin_{j}.ply")

        pcd0 = o3d.io.read_point_cloud(file0)
        pcd1 = o3d.io.read_point_cloud(file1)

        xyz0 = np.asarray(pcd0.points).astype(np.float32)
        xyz1 = np.asarray(pcd1.points).astype(np.float32)

        return sname, xyz0, xyz1, T_gt


class ThreeDLoMatchTestDataset(ThreeDMatchTestDataset):
    """3DLoMatch test dataset"""

    SPLIT_FILES = {"test": "./datasets/splits/test_3dmatch.txt"}
    CONFIG_ROOT = "./datasets/config/3DLoMatch"
