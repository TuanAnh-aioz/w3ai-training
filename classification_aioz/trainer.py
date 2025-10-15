"""
file        : trainer.py
create date : October 15, 2024
author      : truong.manh.le@aioz.io
description : define trainer
"""

import logging
import os
from abc import ABC
from collections import OrderedDict

import torch

from .utils import get_metric, get_optimizer, get_scheduler

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
        self.optim_name_fmt = "round_{round_idx}_optimizer.pth"
        self.lr_scheduler_name_fmt = "round_{round_idx}_lr_scheduler.pth"

    # @abstractmethod
    def fit_one_epoch(self):
        # Define in subclass
        # Return two value: loss, metric after calculate
        pass

    # @abstractmethod
    def fit_one_batch(self):
        # Define in subclass
        # Return two value: loss, metric after calculate
        pass

    def fit_batches(self, n_steps):
        global_loss = 0
        global_metric = 0

        for _ in range(n_steps):
            batch_loss, batch_metric = self.fit_one_batch()
            global_loss += batch_loss
            global_metric += batch_metric

        return global_loss / n_steps, global_metric / n_steps

    def fit_epochs(self, n_epochs=1):
        metric = 0
        loss = 0
        for _ in range(n_epochs):
            loss, metric = self.fit_one_epoch()
        return loss, metric

    def _train(self):
        if self.config["hyper_parameter"]["fit_by_epoch"]:
            _, metric = self.fit_epochs(n_epochs=self.config["hyper_parameter"]["local_steps"])
        else:
            _, metric = self.fit_batches(n_steps=self.config["hyper_parameter"]["local_steps"])
        return metric

    def save_model(self, is_best: bool = False) -> str:
        model_dict = {"round": self.round_idx, "model_state": self.model.state_dict(), "best_metric": self.best_metric}
        # save temporal checkpoint
        _round_idx = "best" if is_best else self.round_idx
        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name_fmt.format(round_idx=_round_idx))
        model_path_temp = checkpoint_path + "_temp"
        torch.save(model_dict, model_path_temp)
        # rename to the target model_path
        os.rename(model_path_temp, checkpoint_path)
        return checkpoint_path

    def load_model(self, checkpoint_path: str):
        model_data = torch.load(checkpoint_path, map_location=self.device)
        if self.model is not None and "model_state" in model_data:
            self.model.load_state_dict(model_data.get("model_state", model_data))
        if "round" in model_data and "best_metric" in model_data:
            return model_data["round"], model_data["best_metric"]

    def load_optimizer(self, optim_path: str):
        model_data = torch.load(optim_path, map_location=self.device)
        if hasattr(self, "optimizer") and self.optimizer is not None and "optimizer_state" in model_data:
            self.optimizer.load_state_dict(model_data.get("optimizer_state", model_data))
        if "round" in model_data:
            return model_data["round"]

    def save_optimizer(self):
        model_dict = {
            "round": self.round_idx,
            "optimizer_state": self.optimizer.state_dict(),
        }
        # save temporal optimizer
        optim_path = os.path.join(self.checkpoint_dir, self.optim_name_fmt.format(round_idx=self.round_idx))
        optim_path_temp = optim_path + "_temp"
        torch.save(model_dict, optim_path_temp)
        # rename to the target model_path
        os.rename(optim_path_temp, optim_path)

    def load_lr_scheduler(self, file_path: str):
        model_data = torch.load(file_path, map_location=self.device)
        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None and "lr_scheduler_state" in model_data:
            self.lr_scheduler.load_state_dict(model_data.get("lr_scheduler_state", model_data))
        if "round" in model_data:
            return model_data["round"]

    def save_lr_scheduler(self):
        model_dict = {
            "round": self.round_idx,
            "lr_scheduler_state": self.lr_scheduler.state_dict(),
        }
        # save temporal lr scheduler
        lr_path = os.path.join(self.checkpoint_dir, self.lr_scheduler_name_fmt.format(round_idx=self.round_idx))
        lr_path_temp = lr_path + "_temp"
        torch.save(model_dict, lr_path_temp)
        # rename to the target model_path
        os.rename(lr_path_temp, lr_path)

    def number_parameters_model(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        data_train: object,
        data_val: object,
        config: dict,
        device: str,
        checkpoint_dir: str = "checkpoints",
    ):
        # so important elements of the trainer: model, dataloader
        super().__init__(model, config, device, checkpoint_dir)
        self.config = config
        torch.manual_seed(self.config["hyper_parameter"]["seed"])

        self.setup_model()
        self.setup_pretrained()

        self.train_dataloader = data_train
        self.validator_dataloader = data_val

        # Setting up necessary elements: loss function, optimizer, lr scheduler, ...
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric = get_metric()

        optimizer_cfg = self.config["hyper_parameter"]["optimizer"]
        scheduler_cfg = self.config["hyper_parameter"]["scheduler"]
        self.optimizer = get_optimizer(
            optimizer_cfg["name"], model, lr_initial=optimizer_cfg.get("lr_initial", 1e-3), optimizer_params=optimizer_cfg.get("params", {})
        )
        self.lr_scheduler = get_scheduler(
            scheduler_cfg["name"],
            self.optimizer,
            scheduler_params=scheduler_cfg.get("params", {}),
            num_epochs=self.config["hyper_parameter"].get("n_rounds", None),
            num_steps_per_epoch=len(self.train_dataloader),
        )

        self.best_weight_path = None

    def setup_pretrained(self):
        pretrained_cfg = self.config["model"].get("pretrained", {})
        pretrained_type = pretrained_cfg.get("type", None)
        pretrained_path = pretrained_cfg.get("path", None)

        if pretrained_type == "resume":
            self.resume_checkpoint(pretrained_path)
        elif pretrained_type == "finetune":
            self.load_pretrained_model(pretrained_path)
        else:
            logger.info(f"Unknown pretrained type: {pretrained_type}. Skipping loading pretrained weights.")

    def resume_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            logger.info(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        # self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        # self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])
        self.round_idx = checkpoint.get("round", 0)
        self.best_metric = checkpoint.get("best_metric", 0.0)

        logger.info(f"Resumed training from round {self.round_idx} | best metric = {self.best_metric:.4f}")

    def load_pretrained_model(self, pretrained_path):
        if not pretrained_path or not os.path.exists(pretrained_path):
            logger.info(f"Pretrained model not found: {pretrained_path}")
            return

        checkpoint = torch.load(pretrained_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained weights from {pretrained_path}")
        if missing_keys:
            logger.info(f"Missing keys: {missing_keys}")

        if unexpected_keys:
            logger.info(f"Unexpected keys: {unexpected_keys}")

    # TODO: setting up model
    def setup_model(self):
        self.model.to(self.device)

    # TODO: Modify this function.
    def fit_one_epoch(self):
        # Example
        epoch_loss = 0
        epoch_metric = 0

        self.model.train()

        for batch_idx, (input, ground_truth, _) in enumerate(self.train_dataloader):
            input = input.to(self.device)
            ground_truth = ground_truth.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(input)
            loss, predictions = self.compute_loss(output=output, ground_truth=ground_truth)
            metric = self.metric(predictions, ground_truth)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_metric += metric
            if batch_idx % self.config["hyper_parameter"]["log_interval"] == 0:
                logger.info(
                    "Epoch {}: [{}/{} ({:.0f}%)]".format(
                        self.round_idx, batch_idx * len(input), len(self.train_dataloader.dataset), 100.0 * batch_idx / len(self.train_dataloader)
                    )
                )

        epoch_loss = epoch_loss / len(self.train_dataloader)
        epoch_metric = epoch_metric / len(self.train_dataloader)
        logger.info("Epoch {}: Train Loss: {:.6f} | Train Metric: {:.6f}".format(self.round_idx, epoch_loss, epoch_metric))

        # Consider lr_scheduler to be similar to the MNIST original train
        self.lr_scheduler.step()
        return epoch_loss, epoch_metric

    # TODO: Modify this function.
    def fit_one_batch(self, update=True):
        # Example
        self.model.train()

        x, ground_truth = next(iter(self.train_dataloader))
        x = x.to(self.device)
        ground_truth = ground_truth.to(self.device)

        self.optimizer.zero_grad()

        output = self.model(x)  # Changed input to x to match the variable name
        loss, predictions = self.compute_loss(output=output, ground_truth=ground_truth)

        metric = self.metric(predictions, ground_truth)

        loss.backward()

        if update:
            self.optimizer.step()
            self.lr_scheduler.step()

        batch_loss = loss.item()
        batch_metric = metric

        return batch_loss, batch_metric

    # TODO: implement this function
    def compute_loss(self, output, ground_truth):
        # output has more than one result. Classification: InceptionV3, GoogLeNet
        if hasattr(self.model, "aux_logits") and self.model.aux_logits:
            # InceptionV3
            if len(output) == 2:
                predictions, aux = output
                loss = self.config["hyper_parameter"]["coeff"] * (self.criterion(predictions, ground_truth) + 0.1 * self.criterion(aux, ground_truth))
            # GoogLeNet
            elif len(output) == 3:
                predictions, aux2, aux1 = output
                loss = self.config["hyper_parameter"]["coeff"] * (
                    self.criterion(predictions, ground_truth) + 0.1 * self.criterion(aux2, ground_truth) + 0.1 * self.criterion(aux1, ground_truth)
                )
            else:
                raise ValueError("The model does not support <aux_logits>")
        else:
            # Image Segmentation case.
            if isinstance(output, OrderedDict):
                # ground_truth = torch.unsqueeze(ground_truth, 1)
                losses = {}
                for name, x in output.items():
                    losses[name] = self.criterion(x, ground_truth)
                if len(losses) == 1:
                    loss = losses["out"]
                else:
                    loss = losses["out"] + 0.5 * losses["aux"]
                predictions = output["out"]
            else:
                predictions = output
                loss = self.config["hyper_parameter"]["coeff"] * self.criterion(predictions, ground_truth)
        return loss, predictions

    def train(self):
        n_rounds = self.config["hyper_parameter"]["n_rounds"]
        for self.round_idx in range(self.round_idx, n_rounds):
            _ = self._train()
            _, val_metric = self.validate_one_epoch()
            if val_metric > self.best_metric:
                self.best_metric = val_metric
                wp = self.save_model(is_best=True)
                self.best_weight_path = wp

            self.save_model()
            # self.save_optimizer()
            # self.save_lr_scheduler()
            self.round_idx += 1

        return self.best_weight_path, self.best_metric

    def validate_one_epoch(self):
        self.model.eval()
        val_loss = 0
        val_metric = 0
        with torch.no_grad():
            for input, ground_truth, _ in self.validator_dataloader:
                input = input.to(self.device)
                ground_truth = ground_truth.to(self.device)

                output = self.model(input)
                loss, predictions = self.compute_loss(output=output, ground_truth=ground_truth)

                metric = self.metric(predictions, ground_truth)

                val_loss += loss.item()
                val_metric += metric

        val_loss /= len(self.validator_dataloader)
        val_metric /= len(self.validator_dataloader)
        logger.info(f"Epoch {self.round_idx}: Validation Loss: {val_loss:.6f} | Validation Metric: {val_metric:.6f}")
        return val_loss, val_metric


class Inference(BaseTrainer):
    def __init__(self, model, test_data, model_weight_path, type, config, device):
        super().__init__(model, test_data, config, device)
        self.type = type
        self.config = config
        self.device = device
        self.load_model(model_weight_path)
        self.__setup_model__()
        self.test_dataloader = test_data

    def __setup_model__(self):
        """Setup device and model mode."""
        self.model.to(self.device)
        self.model.eval()

    def run(self):
        labels = self.config["dataset"]["classes"]
        all_results = []
        with torch.no_grad():
            for input, _, img_paths in self.test_dataloader:
                input = input.to(self.device)
                output = self.model(input)
                _, predicted_classes = torch.max(output, 1)

                for pred, path in zip(predicted_classes, img_paths):
                    label = labels.get(str(pred.item()))
                    if label:
                        all_results.append((os.path.abspath(path), label))

        return all_results
