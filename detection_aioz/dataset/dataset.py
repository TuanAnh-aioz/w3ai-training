"""
file        : dataset.py
create date : October 15, 2024
author      : truong.manh.le@aioz.io
description : define dataset and dataloader
"""

import logging
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import InterpolationMode

from .transforms import SimpleCopyPaste, get_transform
from .utils import GroupedBatchSampler, create_aspect_ratio_groups, load_image_opencv, read_csv_file

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, data_dir: str, config: dict, is_train: bool):
        self.config = config
        self.data_dir = data_dir
        self.metadata_path = os.path.join(self.data_dir, "metadata.csv")
        self.is_train = is_train

        # self.label_map = {v: k for k, v in config["classes"].items()}
        self.label_map = config["dataset"]["classes"]
        if is_train:
            self.transform = get_transform(self.config["dataset"]["transform"]["train"])
        else:
            self.transform = get_transform(self.config["dataset"]["transform"]["validator"])

        self.images_data = defaultdict(lambda: {"bboxes": [], "labels": []})
        self.unique_img_paths = []

        self.load_data()

    def load_data(self):
        logger.debug(f"Loading metadata from: {self.metadata_path}")
        df = read_csv_file(self.metadata_path).dropna()

        for _, row in df.iterrows():
            im_path = os.path.join(self.data_dir, row["image_path"])

            xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            bbox = [xmin, ymin, xmax, ymax]
            label = int(self.label_map.get(row["label"], 0))

            if im_path not in self.images_data:
                self.unique_img_paths.append(im_path)
                if "width" in row and "height" in row:
                    self.images_data[im_path]["width"] = int(row["width"])
                    self.images_data[im_path]["height"] = int(row["height"])

            self.images_data[im_path]["bboxes"].append(bbox)
            self.images_data[im_path]["labels"].append(label)

    def __len__(self):
        return len(self.unique_img_paths)

    def __getitem__(self, index):
        pass


class ImageDetectionDataset(ImageDataset):
    def __init__(self, dataset_dir: str, config: dict, is_train: bool):
        super().__init__(dataset_dir, config, is_train)

    def __getitem__(self, index: int):
        img_path = self.unique_img_paths[index]
        data = self.images_data[img_path]

        img, _ = load_image_opencv(img_path, color=self.config["dataset"]["color"])
        # wt, ht = img.shape[1], img.shape[0]
        # size_org = (wt, ht)

        bboxes = data["bboxes"]
        labels = data["labels"]

        # Apply transform
        if self.transform is not None:
            img = self.transform(img)

        # Convert to tensors
        bboxes = [[float(x) for x in box] for box in bboxes]
        labels = [int(item) for item in labels]
        boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            "image_id": torch.tensor([index]),
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }
        return img, target, img_path

    def get_height_and_width(self, index: int):
        img_path = self.unique_img_paths[index]
        data = self.images_data[img_path]
        if "height" in data and "width" in data:
            return data["height"], data["width"]

        img, _ = load_image_opencv(img_path, color=self.config["dataset"]["color"])
        h, w = img.shape[:2]
        data["height"], data["width"] = h, w
        return h, w


def get_dataloader(dataset_dir: str, config: dict, is_train: bool = True):
    """
    Creates dataloaders for training, validation, or testing.

    Args:
        dataset_dir (str): Path to the dataset directory.
        config (dict): Configuration dictionary containing dataloader parameters.
            Expected keys: 'batch_size', 'num_workers', 'pin_memory'.
        is_train (bool): Whether to prepare data for training (True) or testing (False). Defaults to True.

    Returns:
        tuple: A tuple containing the dataloader and optionally a validation dataloader (None if `is_train` is False).
    """

    full_dataset = ImageDetectionDataset(dataset_dir, config, is_train=is_train)
    dataset_size = len(full_dataset)
    train_size = int(config["dataset"]["split"]["train"] * dataset_size)
    val_size = int(config["dataset"]["split"]["validate"] * dataset_size)

    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_dataset = torch.utils.data.Subset(ImageDetectionDataset(dataset_dir, config, is_train=True), train_indices)
    val_dataset = torch.utils.data.Subset(ImageDetectionDataset(dataset_dir, config, is_train=False), val_indices)
    test_dataset = torch.utils.data.Subset(ImageDetectionDataset(dataset_dir, config, is_train=False), test_indices)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)

    if config["dataset"]["aspect_ratio_group_factor"] >= 0:
        group_ids = create_aspect_ratio_groups(train_dataset, k=config["dataset"]["aspect_ratio_group_factor"])
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, config["dataset"]["batch_size"])
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, config["dataset"]["batch_size"], drop_last=True)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=1,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    return tuple(zip(*batch))


def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*collate_fn(batch))
