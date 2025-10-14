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
from PIL import Image

from .transforms import Compose, CVResize, Normalize, NumpyToTensor, RandomHorizontalFlip


def read_csv_file(file_path: str) -> pd.DataFrame:
    """Reads a CSV file and returns its contents as a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the CSV data.
    """
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
    """Randomly flips the image horizontally and adjusts the steering angle.

    Args:
        image (np.ndarray): The input image.
        steering_angle (float): The steering angle associated with the image.

    Returns:
        tuple[np.ndarray, float]: The flipped image and the adjusted steering angle.
    """

    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def augment(image, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    return image, steering_angle


def load_img(path: str, grayscale: bool = False, target_size: tuple[int, int] = None, crop_size: tuple[int, int] = None) -> np.ndarray:
    """Loads an image with optional resizing and cropping.

    Args:
        path (str): Path to the image file.
        grayscale (bool, optional): Whether to load the image as grayscale. Defaults to False.
        target_size (tuple[int, int], optional): Target size as (width, height). Defaults to None.
        crop_size (tuple[int, int], optional): Crop size as (width, height). Defaults to None.

    Returns:
        np.ndarray: The loaded and processed image as a numpy array.
    """
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
    """Crops the image centered in width and starting from the bottom in height.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        crop_width (int, optional): The width of the crop. Defaults to 150.
        crop_height (int, optional): The height of the crop. Defaults to 150.

    Returns:
        np.ndarray: The cropped image.
    """
    half_the_width = int(img.shape[1] / 2)
    img = img[img.shape[0] - crop_height : img.shape[0], half_the_width - int(crop_width / 2) : half_the_width + int(crop_width / 2)]

    return img


def get_transform(width: int, height: int, use_ImageNetNormalize: bool, train: bool = True, resize: bool = True) -> object:
    """Returns a composed set of transformations for image preprocessing.

    Args:
        width (int): The target width for resizing the image.
        height (int): The target height for resizing the image.
        use_ImageNetNormalize (bool): Whether to apply ImageNet normalization.
        train (bool, optional): Whether the transformations are for training (includes random flips). Defaults to True.
        resize (bool, optional): Whether to resize the image. Defaults to True.

    Returns:
        object: A composed transform pipeline with a list of transformations.
    """
    # Define a list to hold all transformations
    transform_list = []
    # Add resizing if specified
    if resize:
        transform_list.append(CVResize(width=width, height=height))

    # Convert Numpy array to Tensor
    transform_list.append(NumpyToTensor())

    # Add random horizontal flip for data augmentation during training
    if train:
        transform_list.append(RandomHorizontalFlip(0.5))

    # Apply ImageNet normalization if specified
    if use_ImageNetNormalize:
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_list.append(normalize)

    # Return the composed transformations
    return Compose(transform_list)


def load_image_opencv(img_path: str, color: bool = True) -> np.ndarray:
    """Loads an image from a given file path using OpenCV.

    Args:
        img_path (str): The path to the image file.
        color (bool, optional): Whether to load the image in color (True) or grayscale (False). Defaults to True.

    Returns:
        np.ndarray: The loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If the image cannot be loaded from the specified path.

    """
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
