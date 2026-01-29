import logging
import os
import sys
import math
import time
import json
import hydra

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import safetensors

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_pt_utils import get_parameter_names
from transformers import AutoConfig, AdamW, EsmModel
from torch.optim import Adam
import torch.nn.functional as F
import deepspeed


from sklearn.metrics import f1_score
from torchmetrics.regression import R2Score, PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics.classification import BinaryAccuracy, AUROC, BinaryF1Score, BinaryMatthewsCorrCoef, ConfusionMatrix

from biotite.database import rcsb
from typing import List
from tmtools import tm_align
from esm.utils.structure.affine3d import build_affine3d_from_coordinates
from esm.utils.structure.affine3d import Affine3D

import time


def calculate_multiclass_clf_metric(logits, targets):

    from torchmetrics.classification import MulticlassAccuracy, AUROC, MulticlassF1Score, MulticlassMatthewsCorrCoef

    device = logits.device
    
    logits, targets = logits.to(device), targets.to(device)
    num_classes = logits.shape[-1]
    acc_func = MulticlassAccuracy(num_classes).to(device) # macro as default
    acc = acc_func(logits, targets).float()

    auroc_func = AUROC(task="multiclass", num_classes=num_classes).to(device)
    auroc = auroc_func(logits, targets).float()

    f1_score_func = MulticlassF1Score(num_classes=num_classes).to(device) # macro as default
    f1_score = f1_score_func(logits, targets).float()

    mcc_func = MulticlassMatthewsCorrCoef(num_classes=num_classes).to(device)
    mcc = mcc_func(logits, targets).float()

    return {
        "accuracy": acc,
        "auroc": auroc,
        "f1_score": f1_score,
        "mcc": mcc
    }    

def calculate_binary_clf_metric(logits, targets):
    # logits: (M, )
    # targets: (M, )

    device = logits.device
    acc_func = BinaryAccuracy().to(device)
    acc = acc_func(logits, targets)

    auroc_func = AUROC(task="binary").to(device)
    auroc = auroc_func(logits, targets)

    f1_score_func = BinaryF1Score().to(device)
    f1_score = f1_score_func(logits, targets)

    mcc_func = BinaryMatthewsCorrCoef().to(device)
    mcc = mcc_func(logits, targets)

    cf_func = ConfusionMatrix(task="binary", num_classes=2, normalize="true").to(device)
    cf_score = cf_func(logits, targets)
    cf_all_func = ConfusionMatrix(task="binary", num_classes=2, normalize="all").to(device) # normalize to all samples
    cf_all_score = cf_all_func(logits, targets)

    return {
        "accuracy": acc,
        "auroc": auroc,
        "f1_score": f1_score,
        "mcc": mcc,
        "true_neg": cf_score[0, 0],
        "false_pos": cf_score[0, 1],
        "false_neg": cf_score[1, 0],
        "true_pos": cf_score[1, 1],
        "true_neg_toall": cf_all_score[0,0],
        "false_pos_toall": cf_all_score[0, 1],
        "false_neg_toall": cf_all_score[1, 0],
        "true_pos_toall": cf_all_score[1, 1],
    }

    
def calculate_regression_metric(logits, targets):
    # logits: (M, )
    # targets: (M, )

    device = logits.device
    r2score_func = R2Score().to(device)
    r2 = r2score_func(logits, targets)

    pearson_func = PearsonCorrCoef().to(device)
    pearsonr = pearson_func(logits, targets)

    spearman_func = SpearmanCorrCoef().to(device)
    spearmanr = spearman_func(logits, targets)
    return {
        "r2": r2,
        "pearsonr": pearsonr,
        "spearmanr": spearmanr
    }

def calculate_tm_rmsd_score(mobile_chain, target_chain):

    # get C-alpha coordinates
    mobile_coords = torch.tensor(mobile_chain.atom37_positions[..., 1, :]) # L1, 1, 3
    target_coords = torch.tensor(target_chain.atom37_positions[..., 1, :]) # L2, 1, 3
    
    res = tm_align(mobile_coords, target_coords, mobile_chain.sequence, target_chain.sequence)
    return res.tm_norm_chain1, res.tm_norm_chain2, res.rmsd

def pad_structures(items, constant_value=0, dtype=None, truncation_length=600, pad_length=None):
    """Reference to TAPE https://github.com/songlab-cal/tape/blob/6d345c2b2bbf52cd32cf179325c222afd92aec7e/tape/datasets.py#L37
    Always returns torch.Tensor for consistency with PyTorch training.
    """
    batch_size = len(items)
    # Convert numpy arrays and lists to tensors
    if isinstance(items[0], np.ndarray):
        items = [torch.from_numpy(x) for x in items]
    elif isinstance(items[0], List):
        items = [torch.tensor(x) for x in items]

    if pad_length is None:
        shape = [batch_size] + np.max([x.shape for x in items], 0).tolist()
    else:
        shape = [batch_size] + [pad_length]
    if shape[1] > truncation_length:
        shape[1] = truncation_length

    if dtype is None:
        dtype = items[0].dtype

    # Always create torch tensor
    array = torch.full(shape, constant_value, dtype=dtype)

    for arr, x in zip(array, items):
        arrslice = tuple(slice(dim) for dim in x.shape)
        arr[arrslice] = x[:truncation_length]

    return array

def setup_loggings(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info("Configuration args")
    logger.info(cfg)

    # compute
    computed_total_batch_size = (
        cfg.trainer.num_nodes * cfg.optimization.micro_batch_size
    )
    computed_total_batch_size *= torch.cuda.device_count()
    logging.info(
        f"Training with {cfg.trainer.num_nodes} nodes "
        f"micro-batch size {cfg.optimization.micro_batch_size} "
        f"total batch size {computed_total_batch_size} "
        f"and {torch.cuda.device_count()} devices per-node"
    )

    # set save directory path
    cfg.save_dir_path = os.path.join(cfg.trainer.default_root_dir, cfg.run_name)

    return logger


def get_optimizer(optim_groups, optimizer_cfg):

    optim_cls = AdamW if optimizer_cfg.adam_w_mode else Adam

    args = [optim_groups]
    kwargs = {
        "lr": optimizer_cfg.lr,
        "eps": optimizer_cfg.eps,
        "betas": (optimizer_cfg.betas[0], optimizer_cfg.betas[1]),
    }

    optimizer = optim_cls(*args, **kwargs)
    return optimizer

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_ratio: float = 0.0,
    plateau_ratio: float = 0.0,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        min_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum ratio a learning rate would decay to.
        plateau_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The ratio for plateau phase.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        plateau_steps = int(plateau_ratio * num_training_steps)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + plateau_steps:
            return 1.0
        progress = float(current_step - num_warmup_steps - plateau_steps) / float(
            max(1, num_training_steps - num_warmup_steps - plateau_steps)
        )
        return max(
            min_ratio,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def accuracy(logits, labels, ignore_index: int = -100):
    """Adapt from https://github.com/songlab-cal/tape/blob/6d345c2b2bbf52cd32cf179325c222afd92aec7e/tape/models/modeling_utils.py#L623C1-L628C64
    """
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()


def get_dtype(precision):
    """
    Given PTL precision, convert to torch dtype
    """
    if torch.distributed.get_rank() == 0:
        print("precision: ", precision)
    if precision == 16:
        return torch.float16
    elif precision == "bf16-true":
        return torch.bfloat16
    elif precision == "32-true":
        return torch.float32
    else:
        raise NotImplementedError(f"precision {precision} not implemented")


def load_sharded_data(file_path, py_logger=None):
    """Load data from either a single file or sharded format.

    Sharded format:
    - Main file contains metadata: {"shards": [...], "num_samples": N, "sharded": True}
    - Data is loaded from shard files listed in metadata

    Single file format:
    - Standard torch.save format with list of samples

    Returns: list of samples
    """
    from tqdm import tqdm
    import gc

    if py_logger:
        py_logger.info(f"Loading data from {file_path}...")

    # Try loading the file
    data = torch.load(file_path, weights_only=False)

    # Check if it's sharded format
    if isinstance(data, dict) and data.get("sharded", False):
        if py_logger:
            py_logger.info(f"Detected sharded data format with {len(data['shards'])} shards")

        all_data = []
        for shard_path in tqdm(data["shards"], desc="Loading shards"):
            shard_data = torch.load(shard_path, weights_only=False)
            all_data.extend(shard_data)
            del shard_data
            gc.collect()

        if py_logger:
            py_logger.info(f"Done sharded loading.")
        return all_data
    else:
        # Standard single-file format
        return data
