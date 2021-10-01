import argparse
import glob
import os
import time

import gin
import numpy as np
import open3d as o3d
import src.feature
import torch
import tqdm


@gin.configurable()
@torch.no_grad()
def preprocess(feature_class, input_path):
    device = torch.device("cuda")
    feature_extractor = feature_class(device=device)
    feature_name = feature_extractor.__class__.__name__

    elapsed = 0
    files = glob.glob(f"{input_path}/*/*.ply")
    for file in tqdm.tqdm(files):
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)
        start = time.time()
        feats, coords = feature_extractor.extract_feature(points)
        end = time.time()
        elapsed += end - start

        out_name = f"{os.path.splitext(file)[0]}_{feature_name}.pth"
        torch.save(
            dict(feats=feats.cpu(), coords=coords.cpu(), points=points), out_name
        )

    ts_name = os.path.join(input_path, f"{feature_name}.time")
    with open(ts_name, "w") as f:
        f.write(f"total: {elapsed:.4f}, average: {elapsed/len(files):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config file")
    args = parser.parse_args()

    gin.parse_config_file(args.config)

    preprocess()
