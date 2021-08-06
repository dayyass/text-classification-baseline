import ast
import datetime
import logging
import random
import sys
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
    config["path_to_save_logfile"] = config["path_to_save_folder"] / "logging.txt"
    config["path_to_save_target_names_mapping"] = (
        config["path_to_save_folder"] / "target_names.json"
    )

    # tf-idf
    if ("tf-idf" not in config) or (config["tf-idf"] is None):
        config["tf-idf"] = {}
    if "ngram_range" in config["tf-idf"]:
        config["tf-idf"]["ngram_range"] = ast.literal_eval(
            config["tf-idf"]["ngram_range"]
        )

    # logreg
    if ("logreg" not in config) or (config["logreg"] is None):
        config["logreg"] = {}

    return config


def get_logger(path_to_logfile: str) -> logging.Logger:
    """
    Get logger.

    :param str path_to_logfile: path to logfile.
    :return: logger.
    :rtype: logging.Logger
    """

    logger = logging.getLogger("text-clf")
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
