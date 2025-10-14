from enum import Enum


class PlatformTasks(str, Enum):
    linux = "linux"
    window = "window"
    macApple = "macApple"
    macIntel = "macIntel"


class Tasks(str, Enum):
    detection = "detection"
    classification = "classification"
    segmentation = "segmentation"
