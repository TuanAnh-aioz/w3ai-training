import logging
import os
import traceback
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .utils import get_transform, load_image_opencv, read_csv_file

logger = logging.getLogger(__name__)


class ImageDataset(Dataset, ABC):
    def __init__(self, data_dir: str, config: dict, is_train: bool):
        self.config = config
        self.data_dir = data_dir
        self.metadata_path = os.path.join(self.data_dir, "metadata.csv")
        self.transform = get_transform(config, is_train)
        self.img_path = []
        self.ground_truth = []
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = read_csv_file(self.metadata_path)
        if df is None or df.empty:
            raise ValueError(f"Metadata file is empty or unreadable: {self.metadata_path}")

        df = df.dropna()
        for _, row in df.iterrows():
            im_path = os.path.join(self.data_dir, row["file"])
            if not os.path.exists(im_path):
                logger.warning(f"Missing image: {im_path}")
                continue
            self.img_path.append(im_path)
            self.ground_truth.append(int(row["label_index"]))

        logger.info(f"Loaded {len(self.img_path)} samples from {self.metadata_path}")

    def __len__(self):
        return len(self.img_path)

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses must implement __getitem__().")


class ImageClassificationDataset(ImageDataset):
    def __init__(self, dataset_dir: str, config: dict, is_train: bool):
        super(ImageClassificationDataset, self).__init__(dataset_dir, config, is_train)

    def __getitem__(self, index: int):
        img_path = self.img_path[index]
        img = load_image_opencv(img_path, color=self.config["color"])
        if img is None:
            logger.warning(f"Cannot load image: {img_path}")
            img = np.zeros((self.config["height"], self.config["width"], 3), dtype=np.uint8)

        if self.transform is not None:
            img = self.transform(img)

        ground_truth = int(self.ground_truth[index])
        return img, ground_truth, img_path


def get_dataloader(dataset_dir: str, config: dict, is_train=True):
    try:
        dataset = ImageClassificationDataset(dataset_dir, config, is_train=True)
        total_size = len(dataset)
        train_size = int(config["train_split"] * total_size)
        val_size = int(config["val_split"] * (total_size - train_size))
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(config.get("seed", 42))
        subsets = random_split(dataset, [train_size, val_size, test_size], generator=generator)

        transforms = {"train": get_transform(config, train=True), "eval": get_transform(config, train=False)}

        loaders = {}
        if is_train:
            subsets[0].dataset.transform = transforms["train"]
            subsets[1].dataset.transform = transforms["eval"]
            subsets[2].dataset.transform = transforms["eval"]
            names = ["train", "val", "test"]
        else:
            subsets = [dataset]
            dataset.transform = transforms["eval"]
            names = ["test"]

        for i, name in enumerate(names):
            loaders[name] = DataLoader(
                subsets[i],
                shuffle=(name == "train"),
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
                drop_last=(name == "train"),
            )

        return loaders.get("train"), loaders.get("val"), loaders.get("test")

    except Exception as e:
        logger.warning(f"Error creating dataloader: {e}\n{traceback.format_exc()}")
        return None, None, None
