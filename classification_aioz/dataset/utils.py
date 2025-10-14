"""
file        : utils.py
create date : October 15, 2024
author      : truong.manh.le@aioz.io
description : dataset utils
"""

import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode


def read_csv_file(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at [{file_path}]")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file at [{file_path}] is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"Failed to parse the CSV file at [{file_path}].")


def random_flip(image: np.ndarray, steering_angle: float) -> tuple[np.ndarray, float]:

    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):

    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def augment(image, steering_angle, range_x=100, range_y=10):
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    return image, steering_angle


def load_img(path: str, grayscale: bool = False, target_size: tuple[int, int] = None, crop_size: tuple[int, int] = None) -> np.ndarray:
    img = cv2.imread(path)
    if grayscale:
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if target_size:
        if (img.shape[0], img.shape[1]) != target_size:
            img = cv2.resize(img, target_size)

    if crop_size:
        img = central_image_crop(img, crop_size[0], crop_size[1])

    if grayscale:
        img = img.reshape((1, img.shape[0], img.shape[1]))

    return np.asarray(img, dtype=np.float32)


def central_image_crop(img: np.ndarray, crop_width: int = 150, crop_height: int = 150) -> np.ndarray:
    half_the_width = int(img.shape[1] / 2)
    img = img[img.shape[0] - crop_height : img.shape[0], half_the_width - int(crop_width / 2) : half_the_width + int(crop_width / 2)]

    return img


def get_transform(config: dict, train: bool = True):
    width = config.get("width", 224)
    height = config.get("height", 224)
    transforms_cfg = config["transforms"]["train" if train else "val"]

    transform_list = []

    # Convert numpy -> PIL (important if using OpenCV loader)
    transform_list.append(T.ToPILImage())

    # --- Resize ---
    if transforms_cfg.get("resize", True):
        if train and transforms_cfg.get("random_crop", False):
            resize_scale = transforms_cfg.get("resize_scale", 1.15)
            transform_list.extend(
                [
                    T.Resize((int(height * resize_scale), int(width * resize_scale)), interpolation=InterpolationMode.BICUBIC),
                    T.RandomCrop((height, width)),
                ]
            )
        else:
            transform_list.append(T.Resize((height, width), interpolation=InterpolationMode.BICUBIC))

    # --- Data Augmentation (only for train) ---
    if train:
        if transforms_cfg.get("random_horizontal_flip", False):
            transform_list.append(T.RandomHorizontalFlip(p=0.5))

        color_jitter = transforms_cfg.get("color_jitter", None)
        if color_jitter:
            transform_list.append(T.ColorJitter(**color_jitter))

        if "random_rotation" in transforms_cfg:
            transform_list.append(T.RandomRotation(degrees=transforms_cfg["random_rotation"]))

        random_affine = transforms_cfg.get("random_affine", None)
        if random_affine:
            transform_list.append(T.RandomAffine(degrees=0, **random_affine))

    # --- To Tensor ---
    transform_list.append(T.ToTensor())

    # --- Normalize ---
    normalize_type = transforms_cfg.get("normalize", "imagenet").lower()
    if normalize_type == "imagenet":
        transform_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif normalize_type == "default":
        transform_list.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    return T.Compose(transform_list)


def load_image_opencv(img_path: str, color: bool = True) -> np.ndarray:

    img = cv2.imread(img_path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Failed to load image [{}]".format(img_path))
    # Converting BGR color to RGB color format
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def inference_input_processing(img_path, transform, color):
    img = load_image_opencv(img_path, color)
    img = transform(img)
    return img


def segmentation_output_postprocess(output):
    # Visualize the segmentation mask
    # mapping colors
    dpi = 100
    image_height, image_width = output.shape
    fig_width = image_width / dpi
    fig_height = image_height / dpi
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(output, cmap="jet")
    plt.axis("off")  # Hide axis
    plt.xticks([])  # Hide x-axis ticks
    plt.yticks([])  # Hide y-axis ticks
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust plot margins to remove any space
    plt.margins(0, 0)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    image = Image.open(buffer)
    return image
