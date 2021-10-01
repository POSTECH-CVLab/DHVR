import argparse
import logging
import os
import time

import gin
import numpy as np
import pytorch_lightning as pl
import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table

import src.data
import src.feature
import src.models
from src.dhvr import DHVR
from src.utils.file import ensure_dir
from src.utils.logger import setup_logger
from src.utils.misc import count_parameters


def print_table(subset_names, stats, rte_ths, rre_ths):
    console = Console()
    table = Table(show_header=True, header_style="bold")

    columns = ["scene", "recall", "rte", "rre", "time"]
    for col in columns:
        table.add_column(col)

    if stats.ndim == 3:
        stats = stats[-1, :, :]

    stats[:, 0] = (stats[:, 1] < rte_ths) * (stats[:, 2] < rre_ths)
    scene_vals = np.zeros((len(subset_names), 4))
    for sid, _ in enumerate(subset_names):
        curr_scene = stats[:, -1] == sid
        if curr_scene.sum() > 0:
            curr_scene_stats = stats[curr_scene]
            success = curr_scene_stats[:, 0] > 0
            recall = success.mean()
            scene_vals[sid][0] = recall
            scene_vals[sid][1:4] = curr_scene_stats[success, 1:4].mean(0)
        else:
            scene_vals[sid] = None

    for sid, vals in zip(subset_names, scene_vals):
        table.add_row(sid, *[f"{v:.4f}" for v in vals])

    success = stats[:, 0] > 0
    recall = success.mean()
    metrics = stats[success, :4].mean(0)
    metrics[0] = recall
    table.add_row("avg", *[f"{m:.4f}" for m in metrics])
    console.print(table)


def rte_rre(T_pred, T_gt, rte_thresh, rre_thresh, eps=1e-16):
    if T_pred is None:
        return np.array([0, np.inf, np.inf])

    rte = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]) * 100
    rre = (
        np.arccos(
            np.clip(
                (np.trace(T_pred[:3, :3].T @ T_gt[:3, :3]) - 1) / 2, -1 + eps, 1 - eps
            )
        )
        * 180
        / np.pi
    )
    return np.array([rte < rte_thresh and rre < rre_thresh, rte, rre])


def run_benchmark(
    data_loader,
    method,
    TE_THRESH,
    RE_THRESH,
    log_interval=100,
):
    tot_num_data = len(data_loader)
    data_loader_iter = iter(data_loader)

    dataset = data_loader.dataset
    subset_names = dataset.subset_names

    stats = np.zeros((tot_num_data, 5))
    stats[:, -1] = -1
    poses = []

    with torch.no_grad():
        for batch_idx in track(range(tot_num_data)):
            batch = data_loader_iter.next()
            sname, xyz0, xyz1, trans = batch[0]
            sid = subset_names.index(sname)
            T_gt = np.linalg.inv(trans)

            start = time.time()
            T = method.register(xyz0, xyz1)
            end = time.time()

            result = rte_rre(T, T_gt, TE_THRESH, RE_THRESH)
            stats[batch_idx, :3] = result
            stats[batch_idx, 3] = end - start
            stats[batch_idx, 4] = sid
            poses.append(T.numpy())

            if batch_idx % log_interval == 0 and batch_idx > 0:
                cur_stats = stats[:batch_idx]
                cur_recall = cur_stats[:, 0].mean() * 100
                cur_rte = cur_stats[cur_stats[:, 0] > 0, 1].mean()
                cur_rre = cur_stats[cur_stats[:, 0] > 0, 2].mean()
                print(
                    f"recall: {cur_recall:.2f}, rte: {cur_rte:.2f}, rre: {cur_rre:.2f}"
                )

    return subset_names, stats, np.stack(poses, axis=0)


@gin.configurable()
def test(
    out_dir,
    run_name,
    checkpoint_path,
    feature_class,
    model_class,
    dataset_class,
    log_interval,
):
    # initialize data loader
    dataset = dataset_class()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn=lambda x: x
    )
    TE_THRESH = dataset.TE_THRESH
    RE_THRESH = dataset.RE_THRESH

    # initialize device
    device = torch.device("cuda")

    # initialize feature extractor
    feature_extractor = feature_class(device=device)

    # initialize refinement model
    refine_model = model_class().to(device)
    ckpt = torch.load(checkpoint_path)

    def remove_prefix(k, prefix):
        return k[len(prefix) :] if k.startswith(prefix) else k

    state_dict = {remove_prefix(k, "model."): v for k, v in ckpt["state_dict"].items()}
    refine_model.load_state_dict(state_dict)
    logging.info(f"Load refine model from checkpoint {checkpoint_path}")
    logging.info(f"number of parameters: {count_parameters(refine_model)}")
    refine_model.eval()
    dhvr = DHVR(
        device=device, feature_extractor=feature_extractor, refine_model=refine_model
    )

    # run benchmark
    subset_names, stats, poses = run_benchmark(
        dataloader,
        method=dhvr,
        TE_THRESH=TE_THRESH,
        RE_THRESH=RE_THRESH,
        log_interval=log_interval,
    )

    # print_table
    print_table(subset_names, stats, TE_THRESH, RE_THRESH)

    # save results
    exp_dir = os.path.join(out_dir, run_name)
    ensure_dir(exp_dir)
    stat_filename = os.path.join(exp_dir, "stats.npz")
    conf_filename = os.path.join(exp_dir, "config.gin")
    np.savez(stat_filename, stats=stats, names=["dhvr"], poses=poses)
    with open(conf_filename, "w") as f:
        f.write(gin.operative_config_str())
    logging.info(f"Saved results to {stat_filename}, {conf_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--run_name", type=str, required=True, help="experiment title")
    parser.add_argument(
        "--load_path", type=str, required=True, help="path to checkpoint"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="experiments",
        help="path to save benchmark results",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    # random seed
    pl.seed_everything(args.seed)

    # setup config and logger
    gin.parse_config_file(args.config)
    setup_logger(args.run_name, args.debug)

    # start test
    test(args.out_dir, args.run_name, args.load_path)
