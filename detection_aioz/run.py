"""
file        : run.py
create date : October 07, 2024
author      : truong.manh.le@aioz.io
description : main func
"""

import logging
import traceback
from typing import Union

import torch
from aioz_ainode_base.trainer.exception import AINodeTrainerException
from aioz_ainode_base.trainer.schemas import IOExample, IOMetadata, TrainerInput, TrainerOutput

from . import utils
from .dataset import get_dataloader
from .model import get_model
from .trainer import Trainer

logger = logging.getLogger(__name__)


def run(input_obj: Union[dict, TrainerInput] = None) -> TrainerOutput:
    try:
        # Write code here
        if isinstance(input_obj, dict):
            input_obj = TrainerInput.model_validate(input_obj)

        # Load config
        config = utils.load_config(input_obj.config)
        use_gpu = config["use_gpu"] and torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        if device == "cuda":
            config["dataset"]["num_workers"] = 1
            config["dataset"]["pin_memory"] = True

        # Create data loader
        train_dataloader, val_dataloader, test_dataloader = get_dataloader(dataset_dir=input_obj.dataset_directory, config=config)

        # Create Trainer
        model, parameters = get_model(config=config)
        checkpoint_dir = input_obj.checkpoint_directory
        output_dir = input_obj.output_directory

        trainer = Trainer(model, parameters, train_dataloader, val_dataloader, test_dataloader, config, device, checkpoint_dir, output_dir)
        total_params, total_trainable_params = trainer.number_parameters_model()
        logger.info(f"Model parameters: {total_params}, trainable parameters: {total_trainable_params}")
        trainer.train()

        # Testing the model
        results = trainer.inference()

        # Create output
        example = [IOExample(input=IOMetadata(data=path, type=str), output=IOMetadata(data=label, type=str)) for path, label in results]

        output_obj = TrainerOutput(weights=trainer.best_weight_path, metrix=trainer.best_metric, examples=example)
        utils.clean_checkpoints(checkpoint_dir)
        return output_obj

    except Exception:
        logger.warning(f"Occur an error {traceback.format_exc()}")
        raise AINodeTrainerException()
