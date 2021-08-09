import logging
import random
import sys
from argparse import ArgumentParser

import numpy as np


def get_argparse() -> ArgumentParser:
    """
    Get argument parser.

    :return: argument parser.
    :rtype: ArgumentParser
    """

    parser = ArgumentParser(prog="text-clf-train")
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=False,
        default="config.yaml",
        help="Path to config",
    )

    return parser


def get_logger(path_to_logfile: str) -> logging.Logger:
    """
    Get logger.

    :param str path_to_logfile: path to logfile.
    :return: logger.
    :rtype: logging.Logger
    """

    logger = logging.getLogger("text-clf-train")
    logger.setLevel(logging.INFO)

    # create handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(path_to_logfile)
    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    # create formatters and add it to handlers
    stream_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(stream_format)
    file_handler.setFormatter(file_format)

    # add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.

    :param int seed: seed.
    """

    random.seed(seed)
    np.random.seed(seed)
