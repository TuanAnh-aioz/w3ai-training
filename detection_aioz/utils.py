import inspect
import json
import os
import random
from collections import defaultdict
from typing import Iterable

import cv2
import numpy as np
import torch
import torchvision


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


def get_optimizer(config: dict, model: Iterable[torch.nn.Parameter]):
    optimizers = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }

    name = config.get("name", "").lower()
    if name not in optimizers:
        raise ValueError(f"Optimizer '{name}' not supported!")

    optimizer_class = optimizers[name]

    valid_params = inspect.signature(optimizer_class).parameters.keys()

    optim_config = {k: v for k, v in config.items() if k in valid_params}

    return optimizer_class(model, **optim_config)


def get_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    schedulers = {
        "steplr": torch.optim.lr_scheduler.StepLR,
        "reducelronplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "cosineannealing": torch.optim.lr_scheduler.CosineAnnealingLR,
        "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
        "linearlr": torch.optim.lr_scheduler.LinearLR,
        "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
    }

    name = config.get("name", "").lower()
    if name not in schedulers:
        raise ValueError(f"Scheduler '{name}' not supported!")

    scheduler_class = schedulers[name]

    valid_params = inspect.signature(scheduler_class).parameters.keys()

    sched_config = {k: v for k, v in config.items() if k in valid_params}

    return scheduler_class(optimizer, **sched_config)


def apply_nms(orig_prediction, iou_thresh: float = 0.5):
    """
    Applies non max supression and eliminates low score bounding boxes.

    Args:
        orig_prediction: the model output. A dictionary containing element scores and boxes.
        iou_thresh: Intersection over Union threshold. Every bbox prediction with an IoU greater than this value
                        gets deleted in NMS.

    Returns:
        final_prediction: Resulting prediction
    """

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction["boxes"], orig_prediction["scores"], iou_thresh)

    # Keep indices from nms
    final_prediction = orig_prediction
    final_prediction["boxes"] = final_prediction["boxes"][keep]
    final_prediction["scores"] = final_prediction["scores"][keep]
    final_prediction["labels"] = final_prediction["labels"][keep]
    return final_prediction


def IOU(box1: list, box2: list) -> float:
    """
    Intersection over Union - IoU
    *------------
    |   (x2min,y2min)
    |   *----------
    |   | ######| |
    ----|------* (x1max,y1max)
        |         |
        ----------

    Args:
        box1: [xmin,ymin,xmax,ymax]
        box2: [xmin,ymin,xmax,ymax]

    Returns:
        iou -> value of intersection over union of the 2 boxes

    """

    # Compute coordinates of intersection
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])

    # calculate area of intersection rectangle
    inter_area = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)

    # calculate boxes areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # compute IoU
    iou = inter_area / float(area1 + area2 - inter_area)
    assert iou >= 0
    return iou


def compute_AP(ground_truth, predictions, iou_thresh: float = 0.5, n_classes: int = 4):
    """
    Calculates Average Precision across all classes.

    Args:
        ground_truth: list with ground-truth objects. Needs to have the following format: [sequence, frame, obj, [xmin, ymin, xmax, ymax], label, score]
        predictions: list with predictions objects. Needs to have the following format: [sequence, frame, obj, [xmin, ymin, xmax, ymax], label, score]
        iou_thresh: IoU to which a prediction compared to a ground-truth is considered right.
        n_classes: number of existent classes

    Returns:
        Average precision for the specified threshold.
    """
    APs = []

    # group gt and predictions by class
    class_gt_map = {c: [gt for gt in ground_truth if gt[2] == c] for c in range(n_classes)}
    class_pred_map = {c: [pred for pred in predictions if pred[2] == c] for c in range(n_classes)}

    for c in range(n_classes):
        gts = class_gt_map[c]
        preds = class_pred_map[c]

        # group gt by image_id
        gt_by_image = defaultdict(list)
        for gt in gts:
            gt_by_image[gt[0]].append({"bbox": gt[1], "matched": False})

        # sort predictions by score
        preds = sorted(preds, key=lambda x: x[3], reverse=True)

        TP = np.zeros(len(preds))
        FP = np.zeros(len(preds))

        for i, pred in enumerate(preds):
            image_id = pred[0]
            pred_box = pred[1]

            max_iou = 0
            max_gt_idx = -1

            # compare with all gt of the same image
            for idx, gt in enumerate(gt_by_image[image_id]):
                iou = IOU(pred_box, gt["bbox"])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = idx

            if max_iou >= iou_thresh:
                if not gt_by_image[image_id][max_gt_idx]["matched"]:
                    TP[i] = 1
                    gt_by_image[image_id][max_gt_idx]["matched"] = True
                else:
                    FP[i] = 1
            else:
                FP[i] = 1

        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)
        n_gt = len(gts)
        if n_gt == 0:
            continue
        recall = TP_cumsum / (n_gt + 1e-6)
        precision = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)

        # add start point
        recall = np.concatenate(([0], recall))
        precision = np.concatenate(([1], precision))

        # AP = area under PR curve
        AP = np.trapz(precision, recall)
        APs.append(AP)

    return sum(APs) / len(APs) if APs else 0


def compute_mAP(ground_truth, predictions, n_classes: int = 4):
    """
    Calls AP computation for different levels of IoUs, [0.5:.05:0.95].

    Args:
        ground_truth: list with ground-truth objects. Needs to have the following format: [sequence, frame, obj, [xmin, ymin, xmax, ymax], label, score]
        predictions: list with predictions objects. Needs to have the following format: [sequence, frame, obj, [xmin, ymin, xmax, ymax], label, score]
        n_classes: number of existent classes.

    Returns:
        mAp and list with APs for each IoU threshold.
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    APs = [compute_AP(ground_truth, predictions, iou_thresh, n_classes) for iou_thresh in iou_thresholds]
    mAP = np.mean(APs)
    return mAP, APs


def generate_colors(classes: int) -> dict:
    """
    Generate random colors for each class.
    Args:
        classes: number of classes

    Returns:
        colors: dictionary of colors for each class
    """
    colors = {}
    for class_id in classes:
        colors[int(class_id)] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return colors


def get_color(label: int, color_map: dict):
    """
    Get color for a given label.
    Args:
        label: class label
        color_map: dictionary of colors for each class
    Returns:
        color: color for the given label
    """
    return color_map.get(int(label), (255, 255, 255))


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


def draw_transparent_box(img, pt1, pt2, color, alpha=0.3):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_label(img, text, pos, color, font_scale=0.5, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos

    cv2.rectangle(img, (x, y - th - 4), (x + tw, y + baseline), color, -1)
    cv2.putText(img, text, (x, y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_rounded_rectangle(img, pt1, pt2, color, thickness=2, r=10):
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), (270), 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
