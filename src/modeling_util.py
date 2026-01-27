import torch
import deepspeed

from util import get_dtype

def model_init_fn(trainer, model_cfg, **model_kwargs):
    """deepspeed-compatible model initialization
    Do not depend on this for loading model state_dict
    Use `ckpt_path` for lightning trainer instead
    """
    from contextlib import nullcontext

    init_dtype = get_dtype(trainer.precision)

    # Check if using DeepSpeed strategy (has config attribute)
    use_deepspeed = hasattr(trainer.strategy, 'config') and trainer.strategy.config is not None

    if use_deepspeed:
        is_zero3 = trainer.strategy.config["zero_optimization"]["stage"] == 3
        context = deepspeed.zero.Init(
            remote_device=trainer.strategy.remote_device,
            pin_memory=True,
            config_dict_or_path=trainer.strategy.config,
            dtype=init_dtype,
            enabled=is_zero3,
        )
    else:
        # DDP or other strategies - no special context needed
        context = nullcontext()

    from model_module import SequenceClassificationModel, ZeroshotProximityModel, ZeroShotCodebookUtilityModel
    from vqvae_model import VQVAEModel

    with context:
        model = eval(model_cfg.class_name)(
            model_cfg,
            **model_kwargs
        )

    return model

