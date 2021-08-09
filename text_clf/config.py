import logging
import os
import sys


def get_logger() -> logging.Logger:
    """
    Get logger.

    :return: logger.
    :rtype: logging.Logger
    """

    logger = logging.getLogger("text-clf-load-config")
    logger.setLevel(logging.INFO)

    # create handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    # create formatters and add it to handlers
    stream_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_format)

    # add handlers to the logger
    logger.addHandler(stream_handler)

    return logger


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
    logger = get_logger()

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
