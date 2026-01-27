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
        strategy = hydra.utils.instantiate(cfg.lightning.strategy)
    else:
        strategy = "ddp" # distributed data parallel
    
    # callbacks
    callbacks = [
        hydra.utils.instantiate(cfg.lightning.callbacks.checkpoint),
        hydra.utils.instantiate(cfg.lightning.callbacks.lr_monitor),
        hydra.utils.instantiate(cfg.lightning.callbacks.progress_bar)
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

    # forbidden restart model
    assert cfg.model.ckpt_path is None
    # set seed before initializing models
    pl.seed_everything(cfg.optimization.seed)

    # set up trainer
    trainer = setup_trainer(cfg)

    # set up data module
    cfg.data.multi_label = cfg.model.multi_label
    cfg.data.is_global_or_local = cfg.model.is_global_or_local
    if cfg.tokenizer == "WrappedOurPretrainedTokenizer":
        assert cfg.tokenizer_pretrained_ckpt_path is not None
        assert cfg.tokenizer_ckpt_name is not None
        tmp_cfg = omegaconf.OmegaConf.load(os.path.join(exc_dir, "./script/config/pretrain.yaml"))["model"]
        tmp_cfg.quantizer.freeze_codebook = True
        tmp_cfg.quantizer._need_init = False
        tmp_cfg.quantizer.use_linear_project = cfg.quantizer_use_linear_project
        tmp_cfg.encoder.d_model = cfg.model_encoder_dmodel
        tmp_cfg.encoder.n_layers = cfg.model_encoder_nlayers
        tmp_cfg.encoder.v_heads = cfg.model_encoder_vheads
        tmp_cfg.quantizer.codebook_size = cfg.quantizer_codebook_size
        tmp_cfg.quantizer.codebook_embed_size = cfg.quantizer_codebook_embed_size
        tmp_cfg.encoder.d_out = cfg.model_encoder_dout
        # Handle MCQ quantizer settings
        if hasattr(cfg, 'quantizer_type'):
            tmp_cfg.quantizer.quantizer_type = cfg.quantizer_type
        if hasattr(cfg, 'quantizer_num_codebooks'):
            tmp_cfg.quantizer.num_codebooks = cfg.quantizer_num_codebooks
        if hasattr(cfg, 'quantizer_normalize_embeddings'):
            tmp_cfg.quantizer.normalize_embeddings = cfg.quantizer_normalize_embeddings

        pretrained_model_cfg = {
            "model_cfg": tmp_cfg,
            "pretrained_ckpt_path": cfg.tokenizer_pretrained_ckpt_path,
            "ckpt_name": cfg.tokenizer_ckpt_name,
        }
    else:
        pretrained_model_cfg = {}

    datamodule = data_module.ProteinDataModule(
        tokenizer_name=cfg.tokenizer,
        tokenizer_device=getattr(cfg, "tokenizer_device", "cpu"),
        seed=cfg.optimization.seed,
        micro_batch_size=cfg.optimization.micro_batch_size,
        data_args=cfg.data,
        py_logger=logger,
        test_only=getattr(cfg, "test_only", False),
        precompute_tokens=getattr(cfg, "precompute_tokens", False),
        tokenizer_kwargs=pretrained_model_cfg
    )
    datamodule.setup()
    
    if not cfg.data.use_continuous:
        # num_tokens is needed only when using single MLP classifier w/ LMs
        cfg.model.num_tokens = datamodule.get_tokenizer().get_num_tokens()
        # d_model is the codebook embedding dimension
        codebook_embed = datamodule.get_codebook_embedding()
        cfg.model.d_model = codebook_embed.shape[1]
    else:
        # For continuous mode, d_model is the encoder output dimension (codebook_embed_size)
        cfg.model.d_model = cfg.quantizer_codebook_embed_size
    cfg.model.use_sequence = cfg.data.use_sequence

    # set up module module
    model = hydra.utils.instantiate(
        cfg.lightning.model_module,
        _recursive_=False,
        model_cfg=cfg.model,
        trainer=trainer,
        py_logger=logger,
        optimizer_cfg=cfg.optimization,
        all_split_names=datamodule.all_split_names,
        codebook_embedding=None if cfg.data.use_continuous else datamodule.get_codebook_embedding()
    )

    # training pipeline
    
    if not getattr(cfg, "validate_only", False) and not getattr(cfg, "test_only", False):
        logger.info("*********** start training ***********\n\n")
        
        trainer.fit(
            model=model, datamodule=datamodule,
            ckpt_path=cfg.model.ckpt_path
        )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        logger.info(f"Saving final model weights to {cfg.save_dir_path}")
        trainer.save_checkpoint(cfg.save_dir_path, weights_only=True)
        logger.info("Finished saving final model")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        trainer.validate(
            model=model, datamodule=datamodule,
            ckpt_path="best",
        )
    else:
        logger.info("*********** start validation ***********\n\n")
        trainer.validate(
            model=model, datamodule=datamodule,
            ckpt_path=cfg.model.ckpt_path
        )

if __name__ == "__main__":
    main()