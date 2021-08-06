import random
from argparse import ArgumentParser
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

    with open(path_to_config, mode="r") as fp:
        config = yaml.safe_load(fp)

    return config


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.

    :param int seed: seed.
    """

    random.seed(seed)
    np.random.seed(seed)
