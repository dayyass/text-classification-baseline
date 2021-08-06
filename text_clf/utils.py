import datetime
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def get_argparse() -> ArgumentParser:
    """
    Get argument parser.

    :return: argument parser.
    :rtype: ArgumentParser
    """

    parser = ArgumentParser(prog="text-clf")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config",
    )

    return parser


def get_config(path_to_config: str) -> Dict[str, Any]:
    """
    Get config.

    :param str path_to_config: path to config.
    :return: config.
    :rtype: Dict[str, Any]
    """

    now = datetime.datetime.now()

    with open(path_to_config, mode="r") as fp:
        config = yaml.safe_load(fp)

    config["path_to_save_folder"] = (
        Path(config["path_to_save_folder"]) / f"model_{now.date()}_{now.time()}"
    )

    config["path_to_config"] = path_to_config
    config["path_to_save_model"] = config["path_to_save_folder"] / "model.joblib"
    config["path_to_save_logfile"] = config["path_to_save_folder"] / "text-clf.log"
    config["path_to_save_target_names_mapping"] = (
        config["path_to_save_folder"] / "target_names.json"
    )

    return config


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.

    :param int seed: seed.
    """

    random.seed(seed)
    np.random.seed(seed)
