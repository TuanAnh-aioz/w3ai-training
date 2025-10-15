"""
file        : model.py
create date : October 15, 2024
author      : truong.manh.le@aioz.io
description : define backbone
"""

import logging

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


AVAILABLE_BACKBONE = {
    "AlexNet": "alexnet",
    "ResNet18": "resnet18",
    "ResNet34": "resnet34",
    "ResNet50": "resnet50",
    "ResNet101": "resnet101",
    "ResNet152": "resnet152",
    "ResNeXt50": "resnext50_32x4d",
    "ResNeXt101": "resnext101_32x8d",
    "WideResNet50": "wide_resnet50_2",
    "WideResNet101": "wide_resnet101_2",
    "VGG11": "vgg11",
    "VGG13": "vgg13",
    "VGG16": "vgg16",
    "VGG19": "vgg19",
    "SqueezeNet": "squeezenet1_0",
    "InceptionV3": "inception_v3",
    "DenseNet121": "densenet121",
    "DenseNet161": "densenet161",
    "DenseNet169": "densenet169",
    "DenseNet201": "densenet201",
    "GoogLeNet": "googlenet",
    "MobileNetV2": "mobilenet_v2",
    "MNASNet": "mnasnet1_0",
    "ShuffleNetV2": "shufflenet_v2_x1_0",
}


class Net(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2))
        self.conv_layer_2 = nn.Sequential(nn.Conv2d(64, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d(2))
        self.conv_layer_3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_features=512 * 3 * 3, out_features=num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x


def freeze_model(model: nn.Module):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False


def modify_fc_layer(model: nn.Module, config: dict):
    if config["backbone_name"] == "SqueezeNet":
        # SqueezeNet
        model.classifier[1] = torch.nn.Conv2d(512, config["num_classes"], kernel_size=1)
        model.num_classes = config["num_classes"]
    else:
        if hasattr(model, "classifier"):
            if isinstance(model.classifier, torch.nn.Sequential):
                # AlexNet / VGG family / MobileNetV2 / MNASNet
                model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, config["num_classes"])
            else:
                # DenseNet family
                model.classifier = torch.nn.Linear(model.classifier.in_features, config["num_classes"])
        elif hasattr(model, "fc"):
            # ResNet ResNeXt WideResNet family / InceptionV3 / GoogLeNet / ShuffleNetV2
            model.fc = torch.nn.Linear(model.fc.in_features, config["num_classes"])


def get_model(config: dict) -> nn.Module:
    backbone_key = config.get("backbone_name")
    backbone_name = AVAILABLE_BACKBONE.get(backbone_key)

    if backbone_name is None:
        logger.warning(f"Invalid backbone '{backbone_key}', using default Net.")
        return Net(num_classes=config["num_classes"])

    # Load pretrained weights or random init
    use_pretrained = config.get("use_imageNet_pretrained_weight", True)
    model_kwargs = {"pretrained": use_pretrained}

    if not use_pretrained:
        model_kwargs["num_classes"] = config["num_classes"]

    model = getattr(models, backbone_name)(**model_kwargs)

    if use_pretrained and config.get("only_finetune_linear_layer", True):
        logger.info("Freeze all layers except the last fc layer")
        freeze_model(model)

    modify_fc_layer(model, config)

    return model
