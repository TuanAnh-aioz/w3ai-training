"""
file        : dataset.py
create date : October 15, 2024
author      : truong.manh.le@aioz.io
description : define dataset and dataloader
"""

import logging
import os
import traceback

import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split

from .utils import get_transform, load_image_opencv, read_csv_file

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, data_dir: str, config: dict, use_ImageNetNormalize: bool, is_train: bool):
        self.config = config
        self.data_dir = data_dir
        self.metadata_path = os.path.join(self.data_dir, "metadata.csv")  # fmt: file, label_index, label_name

        # resize = True when training or evaluating for all tasks, except image segmentation inference
        self.transform = get_transform(config["width"], config["height"], use_ImageNetNormalize, is_train, resize=True)

        self.img_path = np.asarray([])
        self.ground_truth = np.asarray([])

        self.load_data()

    def load_data(self):
        logger.debug(f"Loading metadata from: {self.metadata_path}")
        df = read_csv_file(self.metadata_path)
        df = df.dropna()
        for _, row in df.iterrows():
            im_path = os.path.join(self.data_dir, row["file"])
            ground_truth = int(row["label_index"])
            self.img_path = np.append(self.img_path, im_path)
            self.ground_truth = np.append(self.ground_truth, ground_truth)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        pass


class ImageClassificationDataset(ImageDataset):
    def __init__(self, dataset_dir: str, config: dict, use_ImageNetNormalize: bool, is_train: bool):
        super(ImageClassificationDataset, self).__init__(dataset_dir, config, use_ImageNetNormalize, is_train)

    def __getitem__(self, index: int):
        img_path = self.img_path[index]
        # Must use OpenCV to load an image because it has a TypeScript version.
        img = load_image_opencv(img_path, color=self.config["color"])
        if self.transform is not None:
            img = self.transform(img)

        ground_truth = int(self.ground_truth[index])
        return img, ground_truth, img_path


def get_dataloader(dataset_dir: str, config: dict, use_ImageNetNormalize: bool = False, is_train: bool = True):
    """
    Creates dataloaders for training, validation, or testing.

    Args:
        dataset_dir (str): Path to the dataset directory.
        config (dict): Configuration dictionary containing dataloader parameters.
            Expected keys: 'batch_size', 'num_workers', 'pin_memory'.
        use_imagenet_normalize (bool): Whether to apply ImageNet normalization. Defaults to False.
        is_train (bool): Whether to prepare data for training (True) or testing (False). Defaults to True.

    Returns:
        tuple: A tuple containing the dataloader and optionally a validation dataloader (None if `is_train` is False).
    """

    try:
        full_dataset = ImageClassificationDataset(dataset_dir, config, use_ImageNetNormalize, is_train=True)

        dataset_size = len(full_dataset)
        train_size = int(config["train_split"] * dataset_size)
        remaining_size = dataset_size - train_size
        val_size = int(config["val_split"] * remaining_size)
        test_size = remaining_size - val_size

        if is_train:
            train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
            val_dataset.dataset.transform = get_transform(
                width=config["width"], height=config["height"], use_ImageNetNormalize=use_ImageNetNormalize, train=False
            )
            test_dataset.dataset.transform = get_transform(
                width=config["width"], height=config["height"], use_ImageNetNormalize=use_ImageNetNormalize, train=False
            )

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
                drop_last=True,
            )

            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
                drop_last=False,
            )

            test_loader = DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
                drop_last=False,
            )

            return train_loader, val_loader, test_loader
        else:
            full_dataset.dataset.transform = get_transform(
                width=config["img_size"][0], height=config["img_size"][1], use_ImageNetNormalize=use_ImageNetNormalize, train=False
            )
            test_loader = DataLoader(
                full_dataset,
                shuffle=False,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
                drop_last=True,
            )

            return None, None, test_loader

    except Exception:
        logger.warning(f"Occur an error {traceback.format_exc()}")
        return None, None, None
