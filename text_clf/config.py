import ast
import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def get_config(path_to_config: str) -> Dict[str, Any]:
    """
    Get config.

    :param str path_to_config: path to config.
    :return: config.
    :rtype: Dict[str, Any]
    """

    if not os.path.exists(path_to_config):
        load_default_config()

    with open(path_to_config, mode="r") as fp:
        config = yaml.safe_load(fp)

    config["path_to_save_folder"] = (
        Path(config["path_to_save_folder"])
        / f"model_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
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


def load_default_config(
    path_to_save_folder: str = ".",
    filename: str = "config.yaml",
) -> None:
    """
    Function to load default config.

    :param str path_to_save_folder: path to save folder (default: '.').
    :param str filename: filename (default: 'config.yaml').
    """

    # get logger
    logger = logging.getLogger("text-clf-load-config")
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    stream_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_format)

    logger.addHandler(stream_handler)

    # default config
    path = os.path.join(path_to_save_folder, filename)

    config = [
        "seed: 42",
        "verbose: true",
        "path_to_save_folder: models",
        "",
        "# data",
        "data:",
        "  train_data_path: data/train.csv",
        "  valid_data_path: data/valid.csv",
        "  sep: ','",
        "  text_column: text",
        "  target_column: target_name_short",
        "",
        "# tf-idf",
        "tf-idf:",
        "  lowercase: true",
        "  ngram_range: (1, 1)",
        "  max_df: 1.0",
        "  min_df: 0.0",
        "",
        "# logreg",
        "logreg:",
        "  penalty: l2",
        "  C: 1.0",
        "  class_weight: balanced",
        "  solver: saga",
        "  multi_class: auto",
        "  n_jobs: -1",
    ]

    if os.path.exists(path):
        error_msg = f"Config {path} already exists."

        logger.error(error_msg)
        raise FileExistsError(error_msg)

    else:
        with open(path, mode="w") as fp:
            for line in config:
                fp.write(f"{line}\n")

        logger.info(f"Default config {path} successfully loaded.")
