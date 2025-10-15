"""
file        : dataset.py
create date : October 15, 2024
author      : truong.manh.le@aioz.io
description : define dataset and dataloader
"""

import logging
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset

from .transforms import get_transform
from .utils import GroupedBatchSampler, create_aspect_ratio_groups, load_image_opencv, read_csv_file

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, data_dir: str, config: dict, is_train: bool):
        self.config = config
        self.data_dir = data_dir
        self.metadata_path = os.path.join(self.data_dir, "metadata.csv")
        self.is_train = is_train

        self.label_map = config["dataset"]["classes"]
        transform_cfg = config["dataset"]["transform"]["train" if is_train else "validator"]
        self.transform = get_transform(transform_cfg)

        self.images_data = defaultdict(lambda: {"bboxes": [], "labels": []})
        self.unique_img_paths = []

        self.load_data()

    def load_data(self):
        df = read_csv_file(self.metadata_path).dropna()
        for _, row in df.iterrows():
            im_path = os.path.join(self.data_dir, row["image_path"])
            if im_path not in self.images_data:
                self.unique_img_paths.append(im_path)
                if "width" in row and "height" in row:
                    self.images_data[im_path]["width"] = int(row["width"])
                    self.images_data[im_path]["height"] = int(row["height"])

            bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
            label = int(self.label_map.get(row["label"], 0))
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
        bboxes = [[float(x) for x in box] for box in data["bboxes"]]
        labels = [int(x) for x in data["labels"]]

        if self.transform is not None:
            img = self.transform(img)

        target = {
            "image_id": torch.tensor([index]),
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, target, img_path

    def get_height_and_width(self, index: int):
        data = self.images_data[self.unique_img_paths[index]]
        if "height" in data and "width" in data:
            return data["height"], data["width"]

        img, _ = load_image_opencv(self.unique_img_paths[index], color=self.config["dataset"]["color"])
        h, w = img.shape[:2]
        data["height"], data["width"] = h, w
        return h, w


def get_dataloader(dataset_dir: str, config: dict, is_train: bool = True):
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = ImageDetectionDataset(dataset_dir, config, is_train=is_train)
    n = len(dataset)

    split = config["dataset"]["split"]
    train_r, val_r = split.get("train", 0.8), split.get("validate", 0.1)
    assert train_r + val_r <= 1.0, "Invalid split ratio: train + validate > 1.0"

    idx = torch.randperm(n).tolist()
    n_train, n_val = int(n * train_r), int(n * val_r)
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]

    # Subset
    train_ds, val_ds, test_ds = map(lambda ids: Subset(dataset, ids), (train_idx, val_idx, test_idx))

    # Sampler
    train_sampler, val_sampler, test_sampler = RandomSampler(train_ds), SequentialSampler(val_ds), SequentialSampler(test_ds)

    # Batch config
    bs = config["dataset"].get("batch_size", 8)
    nw = config["dataset"].get("num_workers", 4)
    pm = config["dataset"].get("pin_memory", True)

    arf = config["dataset"].get("aspect_ratio_group_factor", -1)
    if arf > 0:
        group_ids = create_aspect_ratio_groups(train_ds, k=arf)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, bs)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, bs, drop_last=True)

    train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, num_workers=nw, pin_memory=pm, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, sampler=val_sampler, batch_size=bs, num_workers=nw, pin_memory=pm, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, num_workers=nw, pin_memory=pm, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def collate_fn(batch):
    return tuple(zip(*batch))
