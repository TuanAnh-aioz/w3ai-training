"""
file        : model.py
create date : Dec 17, 2024
author      : anh.tuan-pham.phung@aioz.io
description : define model
"""

import logging

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models.detection import _utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.ops import batched_nms, box_iou

logger = logging.getLogger(__name__)

AVAILABLE_MODEL = {
    "Faster RCNN ResNet50 FPN": "fasterrcnn_resnet50_fpn",
    "Faster RCNN ResNet50 FPN v2": "fasterrcnn_resnet50_fpn_v2",
    "Faster RCNN MobileNetV3-Large FPN": "fasterrcnn_mobilenet_v3_large_fpn",
    "SSD300 VGG16": "ssd300_vgg16",
    "SSDLite320 MobileNetV3-Large": "ssdlite320_mobilenet_v3_large",
    "FCOS ResNet50 FPN": "fcos_resnet50_fpn",
    "RetinaNet ResNet50 FPN": "retinanet_resnet50_fpn",
    "RetinaNet ResNet50 FPN v2": "retinanet_resnet50_fpn_v2",
}


class BasicDetectionModel(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module = None):
        super().__init__()

        if backbone is None:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
        else:
            self.backbone = backbone

        self.classification_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.regression_head = nn.Conv2d(64, 4, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor, targets: list = None):
        features = self.backbone(images)
        logits = self.classification_head(features)
        bbox_reg = self.regression_head(features)

        if self.training:
            assert targets is not None, "Targets must be provided during training."
            loss_dict = self.compute_loss(logits, bbox_reg, targets, features.shape[-2:])
            return loss_dict
        else:
            results = self.inference(logits, bbox_reg, images.shape[-2:])
            return results

    def compute_loss(self, logits, bbox_reg, targets, feature_map_size):
        grid_cells = self.create_grid(feature_map_size)

        cls_loss = 0.0
        reg_loss = 0.0

        for i in range(len(targets)):
            target = targets[i]
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            gt_boxes_expanded = gt_boxes.unsqueeze(1).expand(-1, grid_cells.shape[0], -1)

            ious = box_iou(gt_boxes_expanded.reshape(-1, 4), grid_cells.reshape(-1, 4))
            ious = ious.reshape(gt_boxes.shape[0], -1)
            max_ious, max_iou_idx = ious.max(dim=0)
            assigned_labels = gt_labels[max_iou_idx]
            assigned_labels[max_ious < 0.5] = 0
            logits_img = logits[i].permute(1, 2, 0).reshape(-1, self.num_classes)
            bbox_reg_img = bbox_reg[i].permute(1, 2, 0).reshape(-1, 4)

            cls_loss += F.cross_entropy(logits_img, assigned_labels, reduction="mean")

            positive_mask = assigned_labels > 0
            if positive_mask.any():
                assigned_boxes = gt_boxes[max_iou_idx[positive_mask]]
                predicted_boxes = bbox_reg_img[positive_mask]
                reg_loss += F.smooth_l1_loss(predicted_boxes, assigned_boxes, reduction="mean")

        return {"cls_loss": cls_loss / len(targets), "reg_loss": reg_loss / len(targets)}

    def inference(self, logits, bbox_reg, image_size):
        boxes = self.decode_boxes(bbox_reg, image_size)
        scores = torch.softmax(logits, dim=1).permute(0, 2, 3, 1).reshape(logits.shape[0], -1, self.num_classes)

        results = []
        for i in range(logits.shape[0]):
            scores_img = scores[i]
            boxes_img = boxes[i]
            final_boxes = []
            final_labels = []
            final_scores = []

            for cls_id in range(1, self.num_classes):
                cls_scores = scores_img[:, cls_id]
                cls_boxes = boxes_img

                keep = batched_nms(cls_boxes, cls_scores, torch.tensor([cls_id]).repeat(len(cls_boxes)), iou_threshold=0.5)

                final_boxes.append(cls_boxes[keep])
                final_labels.append(torch.full((len(keep),), cls_id, dtype=torch.int64, device=cls_boxes.device))
                final_scores.append(cls_scores[keep])

            results.append({"boxes": torch.cat(final_boxes), "labels": torch.cat(final_labels), "scores": torch.cat(final_scores)})

        return results

    def create_grid(self, feature_map_size):
        h, w = feature_map_size
        y_coords = torch.arange(h, device=self.device)
        x_coords = torch.arange(w, device=self.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        grid = torch.stack([xx, yy, xx + 1, yy + 1], dim=-1).float()
        grid = grid.view(-1, 4)
        return grid

    def decode_boxes(self, bbox_reg, image_size):
        B, C, H, W = bbox_reg.shape
        bbox_reg = bbox_reg.permute(0, 2, 3, 1).reshape(B, -1, 4)

        grid = self.create_grid(bbox_reg.shape[-2:])
        boxes = grid.unsqueeze(0) + bbox_reg
        return boxes

    @property
    def device(self):
        return next(self.parameters()).device


def create_model(config: dict):
    size_ssd = 512
    if config["model_name"] == "fasterrcnn_resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=config["weights"], weights_backbone=config["weights_backbone"])
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])

    elif config["model_name"] == "fasterrcnn_resnet50_fpn_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=config["weights"], weights_backbone=config["weights_backbone"])
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])

    elif config["model_name"] == "fasterrcnn_mobilenet_v3_large_fpn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=config["weights"], weights_backbone=config["weights_backbone"])
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])

    elif config["model_name"] == "ssd300_vgg16":
        model = torchvision.models.detection.ssd300_vgg16(weights=config["weights"], weights_backbone=config["weights_backbone"])
        in_channels = _utils.retrieve_out_channels(model.backbone, (300, 300))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=config["num_classes"])
        model.transform.min_size = (size_ssd,)
        model.transform.max_size = size_ssd

    elif config["model_name"] == "ssdlite320_mobilenet_v3_large":
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=config["weights"], weights_backbone=config["weights_backbone"])
        in_channels = _utils.retrieve_out_channels(model.backbone, (320, 320))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=config["num_classes"])
        model.transform.min_size = (size_ssd,)
        model.transform.max_size = size_ssd

    elif config["model_name"] == "fcos_resnet50_fpn":
        model = torchvision.models.detection.fcos_resnet50_fpn(weights=config["weights"], weights_backbone=config["weights_backbone"])
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(in_channels=256, num_anchors=num_anchors, num_classes=config["num_classes"])
        model.transform.min_size = (256,)
        model.transform.max_size = 256
        for param in model.parameters():
            param.requires_grad = True

    elif config["model_name"] == "retinanet_resnet50_fpn":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=config["weights"], weights_backbone=config["weights_backbone"])
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(in_channels=256, num_anchors=num_anchors, num_classes=config["num_classes"])

    elif config["model_name"] == "retinanet_resnet50_fpn_v2":
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=config["weights"], weights_backbone=config["weights_backbone"])
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(in_channels=256, num_anchors=num_anchors, num_classes=config["num_classes"])

    return model


def get_model(config):
    model = create_model(config["model"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    return model, parameters
