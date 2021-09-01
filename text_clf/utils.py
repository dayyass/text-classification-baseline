import importlib.util
import logging
import random
import sys
from argparse import ArgumentParser
from typing import Any, Dict

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
        required=True,
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


def close_logger(logger: logging.Logger) -> None:
    """
    Close logger.
    Source: https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile

    :param logging.Logger logger: logger.
    """

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.

    :param int seed: seed.
    """

    random.seed(seed)
    np.random.seed(seed)


def get_grid_search_params(grid_search_params_path: str) -> Dict[str, Any]:
    """
    Get grid_search_params from python file.

    :param str grid_search_params_path: python file with grid_search_params.
    :return: grid_search_params.
    :rtype: Dict[str, Any]
    """

    spec = importlib.util.spec_from_file_location(  # type: ignore
        name="hyperparams",
        location=grid_search_params_path,
    )
    hyperparams = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(hyperparams)  # type: ignore

    grid_search_params = hyperparams.grid_search_params  # type: ignore

    return grid_search_params


def prepare_dict_to_print(dict: Dict[str, Any]) -> str:
    """
    Helper function to create pretty string to print dictionary.

    :param Dict[str, Any] dict: arbitrary dictionary.
    :return: pretty string to print dictionary.
    :rtype: str
    """

    sorted_items = sorted(dict.items(), key=lambda x: x[0])

    pretty_string = ""
    for k, v in sorted_items:
        pretty_string += f"{k}: {v}\n"

    return pretty_string
