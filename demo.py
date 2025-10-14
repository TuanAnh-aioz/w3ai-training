import importlib
import os
import sys

from aioz_ainode_base.log import setup_logging
from aioz_ainode_base.trainer.exception import AINodeTrainerException


def main():
    try:
        wd = sys.argv[1]
        input_obj = {
            "working_directory": wd,
            "config": os.path.join(wd, "config.json"),
            "dataset_directory": os.path.join(wd, "dataset"),
            "output_directory": os.path.join(wd, "outputs"),
            "checkpoint_directory": os.path.join(wd, "checkpoints"),
            "pretrained_model_directory": os.path.join(wd, "pretrained_model"),
        }
        setup_logging(log_file=os.path.join(wd, "log.log"))
        module = importlib.import_module("aioz_node_trainer")
        output = module.run(input_obj)
        output.print_to_console()

    except Exception:
        e = AINodeTrainerException()
        print(f"Occur an error: {e}")


if __name__ == "__main__":
    main()
