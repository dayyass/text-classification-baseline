import importlib.util
import random
from argparse import ArgumentParser
from typing import Any, Dict

import numpy as np


def get_argparse() -> ArgumentParser:
    """Get argument parser.

    Returns:
        ArgumentParser: Argument parser.
    """

    parser = ArgumentParser(prog="text-clf-train")
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        help="Path to config",
    )

    return parser


def set_seed(seed: int) -> None:
    """Set seed for reproducibility.

    Args:
        seed (int): Seed.
    """

    random.seed(seed)
    np.random.seed(seed)


def get_grid_search_params(grid_search_params_path: str) -> Dict[str, Any]:
    """Get grid_search_params from python file.

    Args:
        grid_search_params_path (str): Python file with grid_search_params.

    Returns:
        Dict[str, Any]: grid_search_params.
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
    """Helper function to create pretty string to print dictionary.

    Args:
        dict (Dict[str, Any]): Arbitrary dictionary.

    Returns:
        str: Pretty string to print dictionary.
    """

    sorted_items = sorted(dict.items(), key=lambda x: x[0])

    pretty_string = ""
    for k, v in sorted_items:
        pretty_string += f"{k}: {v}\n"

    return pretty_string
