"""
file        : trainer.py
create date : October 15, 2024
author      : truong.manh.le@aioz.io
description : define trainer
"""

import logging
import os
from abc import ABC  # , abstractmethod

import cv2
import torch

from .dataset.transforms import get_infer_transforms
from .dataset.utils import load_image_opencv
from .utils import (
    apply_nms,
    compute_mAP,
    draw_label,
    draw_rounded_rectangle,
    draw_transparent_box,
    generate_colors,
    get_color,
    get_optimizer,
    get_scheduler,
)

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    def __init__(self, model: torch.nn.Module, config: dict, device: str, checkpoint_dir: str, **kwargs):
        self.model = model
        self.device = device
        self.config = config
        self.round_idx = 0
        self.best_metric = 0.0

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Define file name fmt
        self.checkpoint_name_fmt = "round_{round_idx}.pth"

    # @abstractmethod
    def fit_one_epoch(self):
        # Define in subclass
        # Return two value: loss, metric after calculate
        pass

    # @abstractmethod
    def fit_one_batches(self, n_steps):
        # Define in subclass
        pass

    def load_model(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model
        if self.model is not None and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])

        # Load optimizer
        if hasattr(self, "optimizer") and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load lr_scheduler
        if hasattr(self, "lr_scheduler") and "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # Load scaler (AMP training)
        if hasattr(self, "scaler") and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        # Load round index & best metric
        self.round_idx = checkpoint.get("round", 0) + 1
        self.best_metric = checkpoint.get("best_metric", 0.0)

        return self.model

    def _train(self):
        if self.config["hyper_parameter"]["fit_by_epoch"]:
            _, metric = self.fit_epochs(n_epochs=self.config["hyper_parameter"]["local_steps"])
        else:
            _, metric = self.fit_batches(n_steps=self.config["hyper_parameter"]["local_steps"])
        return metric

    def fit_epochs(self, n_epochs=1):
        metric = 0.0
        loss = 0.0
        for _ in range(n_epochs):
            loss, metric = self.fit_one_epoch()
        # get the metric of the last local epoch ~ the end of a round
        return loss, metric

    def fit_batches(self, n_epochs=1):
        metric = 0.0
        loss = 0.0
        for _ in range(n_epochs):
            loss, metric = self.fit_one_batch()

        avg_loss = loss / n_epochs
        return avg_loss, metric

    def save_model(self, is_best: bool = False) -> str:
        model_dict = {
            "round": self.round_idx,
            "model_state": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "best_metric": self.best_metric,
            "scaler": self.scaler.state_dict(),
        }
        # save temporal checkpoint
        _round_idx = "best" if is_best else self.round_idx

        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name_fmt.format(round_idx=_round_idx))
        model_path_temp = checkpoint_path + "_temp"
        torch.save(model_dict, model_path_temp)
        # rename to the target model_path
        os.rename(model_path_temp, checkpoint_path)
        return checkpoint_path

    def number_parameters_model(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_params, total_trainable_params


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        parameters: torch.nn.Parameter,
        train_data: object,
        val_data: object,
        test_data: object,
        config: dict,
        device: str,
        checkpoint_dir: str = "checkpoints",
        output_dir: str = "dataset",
    ):
        # so important elements of the trainer: model, dataloader
        super(Trainer, self).__init__(model, config, device, checkpoint_dir)
        torch.manual_seed(int(self.config["hyper_parameter"]["seed"]))
        self.setup_model()
        self.train_dataloader = train_data
        self.val_dataloader = val_data
        self.test_dataloader = test_data
        self.output_dir = output_dir

        self.optimizer = get_optimizer(config=self.config["hyper_parameter"]["optimizer"], model=parameters)
        self.lr_scheduler = get_scheduler(config=self.config["hyper_parameter"]["lr_scheduler"], optimizer=self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()
        self.best_weight_path = None

        if self.config["model"]["resume"]:
            self.resume()

    def resume(self):
        checkpoint = torch.load(self.config["model"]["resume"], map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])

        self.best_metric = checkpoint.get("best_metric", 0.0)
        self.round_idx = checkpoint.get("round", 0) + 1
        self.best_weight_path = self.config["model"]["resume"]
        logger.info(f"Resumed training from {self.best_weight_path}")

    def setup_model(self):
        self.model.to(self.device)

    def fit_one_epoch(self):
        epoch_loss = 0.0
        self.model.train()
        for batch_idx, (inputs, targets, _) in enumerate(self.train_dataloader):
            images = [input.to(self.device) for input in inputs]
            targets = [{k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast():
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += losses.item()
            if batch_idx % self.config["hyper_parameter"]["log_interval"] == 0 or batch_idx == len(self.train_dataloader) - 1:
                logger.info(
                    "Epoch {}: [{}/{} ({:.0f}%)]".format(
                        self.round_idx, batch_idx, len(self.train_dataloader), 100.0 * batch_idx / len(self.train_dataloader)
                    )
                )

        epoch_loss = epoch_loss / len(self.train_dataloader)
        self.lr_scheduler.step()
        mAP = self.evaluate()
        logger.info("Epoch {}: Train Loss: {:.6f} | Val mAP: {:.6f}".format(self.round_idx, epoch_loss, mAP))

        return epoch_loss, mAP

    def fit_one_batch(self):
        batch_loss_total = 0.0
        mAP = 0.0
        self.model.train()
        # Train
        data_iterator = iter(self.train_dataloader)
        try:
            inputs, targets, _ = next(data_iterator)
        except StopIteration:
            data_iterator = iter(self.train_dataloader)
            inputs, targets, _ = next(data_iterator)

        images = [input.to(self.device) for input in inputs]
        targets = [{k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast():
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        self.optimizer.zero_grad()
        self.scaler.scale(losses).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        batch_loss_total += losses.item()

        # Validator
        mAP = self.evaluate_on_validation()
        return batch_loss_total, mAP

    def evaluate(self):
        self.model.eval()
        ground_truth = []
        predictions = []
        with torch.no_grad():
            for images, targets, _ in self.val_dataloader:
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                outputs = self.model(images)
                for idx in range(len(images)):
                    outputs[idx] = apply_nms(outputs[idx], iou_thresh=self.config["dataset"]["threshold"])

                for out, target in zip(outputs, targets):
                    for i in range(len(target["boxes"])):
                        ground_truth.append(
                            [
                                target["image_id"].detach().cpu().numpy()[0],
                                target["boxes"].detach().cpu().numpy()[i].tolist(),
                                target["labels"].detach().cpu().numpy()[i].item(),
                            ]
                        )

                    for j in range(len(out["boxes"])):
                        predictions.append(
                            [
                                target["image_id"].detach().cpu().numpy()[0],
                                out["boxes"].detach().cpu().numpy()[j].tolist(),
                                out["labels"].detach().cpu().numpy()[j].item(),
                                out["scores"].detach().cpu().numpy()[j].item(),
                            ]
                        )

        mAP, _ = compute_mAP(ground_truth, predictions, n_classes=self.config["model"]["num_classes"])
        return mAP

    def train(self):
        self.start_round = self.round_idx
        n_rounds = self.config["hyper_parameter"]["n_rounds"]
        for _ in range(self.start_round, n_rounds):
            metric = self._train()
            if metric > self.best_metric:
                self.best_metric = metric
                wp = self.save_model(is_best=True)
                self.best_weight_path = wp

            # self.save_model()
            self.round_idx += 1

        return self.best_weight_path, self.best_metric

    def inference(self):
        results = []
        self.load_model(self.best_weight_path)
        self.model.to(self.device)
        self.model.eval()

        os.makedirs(self.output_dir, exist_ok=True)
        color_map = generate_colors(list(self.config["dataset"]["classes"].values()))
        id2class = {v: k for k, v in self.config["dataset"]["classes"].items()}

        with torch.no_grad():
            for _, (images, _, img_paths) in enumerate(self.test_dataloader):
                images_input = [img.to(self.device).float() for img in images]
                predictions = self.model(images_input)

                for i, prediction in enumerate(predictions):
                    _, orig_image = load_image_opencv(img_paths[i], normalize=False)

                    boxes = prediction["boxes"].cpu().numpy().astype(int)
                    labels = prediction["labels"].cpu().numpy()
                    scores = prediction["scores"].cpu().numpy()

                    for box, label, score in zip(boxes, labels, scores):
                        if score > self.config["dataset"]["threshold"] and label != 0:
                            xmin, ymin, xmax, ymax = box
                            label_name = id2class.get(label, "unknown")
                            color = get_color(label, color_map)

                            draw_transparent_box(orig_image, (xmin, ymin), (xmax, ymax), color, alpha=0.2)
                            draw_rounded_rectangle(orig_image, (xmin, ymin), (xmax, ymax), color, 2, r=8)
                            draw_label(orig_image, f"{label_name}: {score:.2f}", (xmin, ymin), color)

                    file_name = os.path.basename(img_paths[i])
                    name, ext = os.path.splitext(file_name)
                    output_image_path = os.path.join(self.output_dir, f"{name}_output{ext}")
                    cv2.imwrite(output_image_path, orig_image)

                    results.append((os.path.abspath(output_image_path), img_paths[i]))

        return results


class Inference(BaseTrainer):
    def __init__(
        self,
        model,
        inference_data,
        model_weight_path,
        config,
        device,
        output_dir: str = "dataset",
    ):
        super().__init__(model, inference_data, config, device)
        self.config = config
        self.device = device
        self.load_model(model_weight_path)
        self.__setup_model__()
        self.inference_dataloader = inference_data
        self.output_dir = output_dir

    def __setup_model__(self):
        """Setup device and model mode."""
        self.model.to(self.device)
        self.model.eval()

    def run(self):
        all_results = []
        color_map = generate_colors(self.config["dataset"]["classes"].values())
        id2class = {v: k for k, v in self.config["dataset"]["classes"].items()}
        with torch.no_grad():
            for img_path in os.listdir(self.inference_dataloader):
                if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                image_name = os.path.splitext(os.path.basename(img_path))[0]
                img_path = os.path.join(self.inference_dataloader, img_path)

                img, orig_image = load_image_opencv(img_path, normalize=False)
                image = get_infer_transforms(img)
                image = torch.unsqueeze(image, 0)

                outputs = self.model(image.to(self.device))
                for prediction in outputs:
                    boxes = prediction["boxes"].cpu().numpy()
                    labels = prediction["labels"].cpu().numpy()
                    scores = prediction["scores"].cpu().numpy()

                    for box, label, score in zip(boxes, labels, scores):
                        if score > self.config["dataset"]["threshold"] and label != 0:
                            xmin, ymin, xmax, ymax = map(int, box)
                            label_name = id2class.get(label, "unknown")
                            color = get_color(label, color_map)

                            draw_transparent_box(orig_image, (xmin, ymin), (xmax, ymax), color, alpha=0.2)
                            draw_rounded_rectangle(orig_image, (xmin, ymin), (xmax, ymax), color, 2, r=8)
                            draw_label(orig_image, f"{label_name}: {score:.2f}", (xmin, ymin), color)

                    output_image_path = os.path.join(self.output_dir, f"{image_name}_output.png")
                    cv2.imwrite(output_image_path, orig_image)

                    all_results.append((img_path, output_image_path))

        return all_results
