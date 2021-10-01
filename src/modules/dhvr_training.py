from test import rte_rre

import gin
import numpy as np
import pytorch_lightning as pl
import src.utils.geometry as geometry
import torch


@gin.configurable()
class DHVRTraining(pl.LightningModule):
    def __init__(
        self,
        dhvr,
        optimizer_class,
        scheduler_class,
        log_every_n_steps,
        load_weights,
        load_optimizers,
        checkpoint_path,
        export_path,
        debug,
        best_metric,
        best_metric_type,
        rte_thresh,
        rre_thresh,
    ):
        super().__init__()
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        self.best_metric_value = -np.inf if best_metric_type == "maximize" else np.inf
        self.model = dhvr.refine_model

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters())
        scheduler = self.scheduler_class(optimizer)

        opts = dict(optimizer=optimizer)
        opts["lr_scheduler"] = dict(scheduler=scheduler, interval="epoch")
        return opts

    def on_pretrain_routine_start(self) -> None:
        if self.trainer.is_global_zero:
            print(self.model)
            print("Optimzers: ", self.optimizers())
            print("Schedulers: ", self.trainer.lr_schedulers)

    def training_step(self, batch, batch_idx):
        pcd0, pcd1, coord0, coord1, feat0, feat1, T_gt = (
            batch["pcd0"],
            batch["pcd1"],
            batch["sinput0_C"],
            batch["sinput1_C"],
            batch["sinput0_F"],
            batch["sinput1_F"],
            batch["T_gt"][0],
        )

        self.dhvr.refine_model.train()
        T, register_time = self.dhvr.register(pcd0, pcd1, coord0, coord1, feat0, feat1)
        hspace = self.dhvr.hspace

        angle_gt = geometry.rotation_to_axis_angle(T_gt[:3, :3])
        index_gt = torch.cat(
            [
                torch.floor(angle_gt / self.dhvr.r_binsize),
                torch.floor(T_gt[:3, 3] / self.dhvr.t_binsize),
            ],
            dim=0,
        ).int()

        index_diff = (hspace.C[:, 1:] - index_gt).abs().sum(dim=1)
        index_min = index_diff.argmin()

        # if the offset is larger than 3 voxels, skip current batch
        if index_diff[index_min].item() > 3:
            return

        target = torch.zeros_like(hspace.F)
        target[index_min] = 1.0

        # criteria = BalancedLoss()
        criteria = torch.nn.BCEWithLogitsLoss()
        loss = criteria(hspace.F, target)

        loss_float = loss.detach().cpu().item()
        success, rte, rre = rte_rre(
            T.cpu().numpy(), T_gt.cpu().numpy(), self.rte_thresh, self.rre_thresh
        )

        values = dict(
            loss=loss_float, success=success, rte=rte, rre=rre, time=register_time
        )
        self.log_dict(values, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.trainer.val_dataloaders[0].dataset.reset_seed(0)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pcd0, pcd1, coord0, coord1, feat0, feat1, T_gt = (
            batch["pcd0"],
            batch["pcd1"],
            batch["sinput0_C"],
            batch["sinput1_C"],
            batch["sinput0_F"],
            batch["sinput1_F"],
            batch["T_gt"][0],
        )

        self.dhvr.refine_model.eval()
        T, register_time = self.dhvr.register(pcd0, pcd1, coord0, coord1, feat0, feat1)
        success, rte, rre = rte_rre(
            T.cpu().numpy(), T_gt.cpu().numpy(), self.rte_thresh, self.rre_thresh
        )

        values = dict(recall=success, rte=rte, rre=rre, time=register_time)
        return values

    @torch.no_grad()
    def validation_epoch_end(self, results) -> None:
        assert len(results) > 0
        out_results = dict()
        for k in ["recall", "rte", "rre", "time"]:
            out_results[f"val/{k}"] = np.stack([r[k] for r in results]).mean(0)

        def compare(prev, cur):
            return prev < cur if self.best_metric_type == "maximize" else prev > cur

        if not self.trainer.running_sanity_check and compare(
            self.best_metric_value, out_results[self.best_metric]
        ):
            self.best_metric_value = out_results[self.best_metric]
            out_results[f"{self.best_metric}_best"] = self.best_metric_value
        self.log_dict(out_results)
