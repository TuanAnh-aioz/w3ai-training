"""
file        : utils.py
create date : October 10, 2024
author      : truong.manh.le@aioz.io
description : support func
"""

import json
import os

import torch


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


def accuracy(preds, y):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc.item()


def get_metric(**kwargs):
    return accuracy


def get_optimizer(optimizer_name, net, lr_initial=1e-3):
    if optimizer_name == "adam":
        return torch.optim.Adam([param for param in net.parameters() if param.requires_grad], lr=lr_initial)

    elif optimizer_name == "sgd":
        return torch.optim.SGD([param for param in net.parameters() if param.requires_grad], lr=lr_initial)

    else:
        raise NotImplementedError("Other optimizer are not implemented")


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
