import logging
import traceback
from typing import Union

import torch
from aioz_ainode_base.trainer.exception import AINodeTrainerException
from aioz_ainode_base.trainer.schemas import IOExample, IOMetadata, TrainerInput, TrainerOutput

from . import utils
from .dataset import get_dataloader
from .model import get_model
from .trainer import Inference, Trainer

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
        train_loader, val_loader, test_loader = get_dataloader(
            dataset_dir=input_obj.dataset_directory,
            config=config["dataset"],
            is_train=True,
        )

        # Create Trainer
        model = get_model(config=config["model"])
        checkpoint_dir = input_obj.checkpoint_directory
        # pretrained_model_dir = input_obj.pretrained_model_directory

        trainer = Trainer(model, train_loader, val_loader, config, device, checkpoint_dir)
        logger.info(f"Model parameters: {trainer.number_parameters_model()}")
        best_checkpoint, best_metric = trainer.train()

        # Inference
        inference = Inference(model, test_loader, best_checkpoint, "image_classification", config, device)
        results = inference.run()

        # # Create output
        example = [IOExample(input=IOMetadata(data=path, type=str), output=IOMetadata(data=label, type=str)) for path, label in results]
        output_obj = TrainerOutput(weights=best_checkpoint, metric=best_metric, examples=example)
        utils.clean_checkpoints(checkpoint_dir)
        return output_obj

    except Exception:
        logger.warning(f"Occur an error {traceback.format_exc()}")
        raise AINodeTrainerException()
