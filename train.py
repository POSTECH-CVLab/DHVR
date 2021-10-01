import argparse
import os

import gin
import pytorch_lightning as pl
import torch

import src.feature
import src.models
from src.data import make_data_loader
from src.dhvr import DHVR
from src.modules import get_training_module
from src.utils.file import ensure_dir
from src.utils.logger import setup_logger
from src.utils.misc import logged_hparams


@gin.configurable()
def train(
    save_path,
    config_path,
    project_name,
    run_name,
    gpus,
    training_module,
    feature_class,
    model_class,
    log_every_n_steps,
    refresh_rate_per_second,
    best_metric,
    max_epoch,
    train_dataset,
    val_dataset,
    num_workers,
    batch_size,
    accumulate_grad_batches,
):
    save_path = os.path.join(save_path, run_name)
    ensure_dir(save_path)

    train_dataloader = make_data_loader(
        train_dataset(),
        phase="train",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dataloader = make_data_loader(
        val_dataset(),
        phase="val",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    device = torch.device("cuda" if gpus > 0 else "cpu")

    feature_extractor = feature_class(device=device)
    feature_extractor.freeze()
    refine_model = model_class().to(device)
    dhvr = DHVR(
        device=device, feature_extractor=feature_extractor, refine_model=refine_model
    )
    pl_module = get_training_module(training_module)(dhvr=dhvr)

    callbacks = [
        pl.callbacks.ProgressBar(refresh_rate=refresh_rate_per_second),
        pl.callbacks.ModelCheckpoint(
            dirpath=save_path, monitor=best_metric, save_last=True, save_top_k=1
        ),
        pl.callbacks.LearningRateMonitor(),
    ]
    gin.finalize()
    hparams = logged_hparams()
    loggers = [
        pl.loggers.WandbLogger(
            name=run_name,
            save_dir=save_path,
            project=project_name,
            log_model=True,
            config=hparams,
        )
    ]
    trainer = pl.Trainer(
        default_root_dir=save_path,
        max_epochs=max_epoch,
        gpus=gpus,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=log_every_n_steps,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # write config file
    with open(os.path.join(save_path, "config.gin"), "w") as f:
        f.write(gin.operative_config_str())

    trainer.fit(
        pl_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--save_path", type=str, default="experiments")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    pl.seed_everything(args.seed)
    gin.parse_config_file(args.config)
    setup_logger(args.run_name, args.debug)

    ensure_dir(args.save_path)

    train(
        save_path=args.save_path,
        config_path=args.config,
        run_name=args.run_name,
        gpus=args.gpus,
    )
