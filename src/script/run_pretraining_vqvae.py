import os
import sys
from glob import glob

import hydra
import omegaconf

import pytorch_lightning as pl
import torch

from torch.utils.tensorboard import SummaryWriter

# local imports
exc_dir = os.path.dirname(os.path.dirname(__file__)) # "src/"
sys.path.append(exc_dir)
exc_dir_baseline = os.path.join(os.path.abspath(exc_dir), "baselines")
all_baseline_names = glob(exc_dir_baseline + "/*")
for name in all_baseline_names:
    if name != "cheap_proteins":
        sys.path.append(os.path.join(exc_dir_baseline, name))
sys.path.append(exc_dir_baseline)


import data_module
from util import setup_loggings


def setup_trainer(cfg):
    trainer_logger = hydra.utils.instantiate(cfg.lightning.logger)
    # strategy determins distributed training
    if cfg.deepspeed_path:
        # deepspeed strategy requires pytorch lightning plugin augmentation
        # to run with zero-2d and internal deepspeed config
        strategy = hydra.utils.instantiate(cfg.lightning.strategy)
    else:
        # distributed data parallel
        strategy = "ddp"
    
    callbacks = [
        hydra.utils.instantiate(cfg.lightning.callbacks.checkpoint),
        hydra.utils.instantiate(cfg.lightning.callbacks.lr_monitor),
        hydra.utils.instantiate(cfg.lightning.callbacks.progress_bar),
    ]
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        plugins=[],
        strategy=strategy,
        logger=trainer_logger,
    )

    return trainer


@hydra.main(version_base=None, config_path="config")
def main(cfg):
    """
    Launch supervised fine-tuning using a hydra config, for protein classification
    """
    omegaconf.OmegaConf.resolve(cfg)

    # set up python loggings
    logger = setup_loggings(cfg)

    # restart model
    if cfg.model.ckpt_path:
        # checkpoint will overwrite state dict load anyway
        logger.info(
            f"Resuming from checkpoint {cfg.model.ckpt_path}. "
        )

    # set seed before initializing models
    pl.seed_everything(cfg.optimization.seed)

    # set up trainer
    trainer = setup_trainer(cfg)

    # set up data module
    datamodule = data_module.PretrainingDataModule(
        device=torch.distributed.get_rank() if torch.distributed.is_initialized() else "cpu",
        seed=cfg.optimization.seed,
        micro_batch_size=cfg.optimization.micro_batch_size,
        data_args=cfg.data,
        py_logger=logger,
        test_only=getattr(cfg, "test_only", False),
    )
    datamodule.setup()

    # set up module module
    model = hydra.utils.instantiate(
        cfg.lightning.model_module,
        _recursive_=False,
        model_cfg=cfg.model,
        trainer=trainer,
        py_logger=logger,
        optimizer_cfg=cfg.optimization,
        all_split_names=datamodule.all_split_names
    )

    # training pipeline
    if not getattr(cfg, "validate_only", False) and not getattr(cfg, "test_only", False):
        logger.info("*********** start training ***********\n\n")
        
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.model.ckpt_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        logger.info(f"Saving final model weights to {cfg.save_dir_path}")
        trainer.save_checkpoint(cfg.save_dir_path, weights_only=True)
        logger.info("Finished saving final model")
        if torch.distributed.is_initialized():
            # assumes deepspeed strategy, set barrier to prevent hang by saving on all ranks
            # Barrier avoids checkpoint corruption if node 0 exits earlier than other
            # nodes, which can trigger worker node termination
            torch.distributed.barrier()
        # Only run validation if val_batches > 0
        if getattr(cfg.trainer, "limit_val_batches", 1) != 0:
            trainer.validate(model=model, datamodule=datamodule, ckpt_path="best")
    else:
        logger.info("*********** start validation ***********\n\n")
        trainer.validate(
            model=model, datamodule=datamodule,
            ckpt_path=cfg.model.ckpt_path
        )


if __name__ == "__main__":
    main()