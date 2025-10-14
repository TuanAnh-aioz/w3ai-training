"""
file        : utils.py
create date : October 10, 2024
author      : truong.manh.le@aioz.io
description : support func
"""

import json
import logging
import os

import torch

logger = logging.getLogger(__name__)


def load_config(fp: str) -> dict:
    """load json config

    Args:
        fp (str): file path, absolute path

    Returns:
        dict: config dictionary
    """
    with open(fp, "r") as file:
        d = json.load(file)
    return d


def accuracy(preds: torch.Tensor, y: torch.Tensor) -> float:
    if preds.device != y.device:
        y = y.to(preds.device)

    _, predicted = torch.max(preds, dim=1)
    correct = (predicted == y).sum()
    acc = correct.float() / y.size(0)
    return acc.item()


def get_metric(**kwargs):
    return accuracy


def get_optimizer(optimizer_name, net, lr_initial=1e-3, optimizer_params=None, param_groups=None):
    optimizer_name = optimizer_name.lower()
    optimizer_params = optimizer_params or {}

    # Choose params
    if param_groups is not None:
        params = param_groups
    else:
        params = [p for p in net.parameters() if p.requires_grad]

    # Optimizer mapping
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "adamax": torch.optim.Adamax,
        "lion": torch.optim.Lion if hasattr(torch.optim, "Lion") else None,  # PyTorch ≥ 2.1
    }

    if optimizer_name not in optimizers or optimizers[optimizer_name] is None:
        raise NotImplementedError(f"Optimizer '{optimizer_name}' is not implemented or not available in this PyTorch version.")

    # Merge lr into optimizer_params
    if "lr" not in optimizer_params:
        optimizer_params["lr"] = lr_initial

    optimizer_class = optimizers[optimizer_name]
    optimizer = optimizer_class(params, **optimizer_params)

    logger.info(f"Using optimizer: {optimizer_name.upper()} with params: {optimizer_params}")
    return optimizer


def get_scheduler(scheduler_name, optimizer, scheduler_params=None, num_epochs=None, num_steps_per_epoch=None):
    scheduler_name = scheduler_name.lower()
    scheduler_params = scheduler_params or {}

    schedulers = {
        "step": torch.optim.lr_scheduler.StepLR,
        "multi_step": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
        "cosine_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "onecycle": torch.optim.lr_scheduler.OneCycleLR,
        "linear": torch.optim.lr_scheduler.LinearLR,
        "polynomial": torch.optim.lr_scheduler.PolynomialLR if hasattr(torch.optim.lr_scheduler, "PolynomialLR") else None,
    }

    if scheduler_name not in schedulers or schedulers[scheduler_name] is None:
        raise NotImplementedError(f"Scheduler '{scheduler_name}' is not implemented or not available in this PyTorch version.")

    # Special handling for some schedulers
    if scheduler_name in ["onecycle"]:
        # For OneCycleLR, you must provide total_steps or epochs * steps_per_epoch
        if num_epochs is None or num_steps_per_epoch is None:
            raise ValueError("For OneCycleLR, you must provide num_epochs and num_steps_per_epoch.")
        total_steps = num_epochs * num_steps_per_epoch
        scheduler_params.setdefault("total_steps", total_steps)
        scheduler_params.setdefault("max_lr", [pg["lr"] for pg in optimizer.param_groups])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
    else:
        scheduler_class = schedulers[scheduler_name]
        scheduler = scheduler_class(optimizer, **scheduler_params)

    logger.info(f"Using LR scheduler: {scheduler_name.upper()} with params: {scheduler_params}")
    return scheduler


def clean_checkpoints(folder: str, keep_best: bool = True):

    if not os.path.isdir(folder):
        raise ValueError(f"{folder} error")

    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue

        if fname.endswith(".pth"):
            if keep_best and fname.endswith("_best.pth"):
                continue
            try:
                os.remove(fpath)
            except Exception as e:
                print(f"{fpath}: {e}")
