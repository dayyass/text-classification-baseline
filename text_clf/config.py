import ast
import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from .lemmatizer import LemmatizerPymorphy2, Preprocessor


def get_config(path_to_config: str) -> Dict[str, Any]:
    """Get config.

    Args:
        path_to_config (str): Path to config.

    Returns:
        Dict[str, Any]: Config.
    """

    with open(path_to_config, mode="r") as fp:
        config = yaml.safe_load(fp)

    # backward compatibility
    if "experiment_name" not in config:
        config["experiment_name"] = "model"

    config["path_to_save_folder"] = (
        Path(config["path_to_save_folder"])
        / f"{config['experiment_name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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

    if "preprocessing" in config:  # backward compatibility
        lemmatization = config["preprocessing"]["lemmatization"]

        if lemmatization:
            if lemmatization == "pymorphy2":
                lemmatizer = LemmatizerPymorphy2()
                preprocessor = Preprocessor(lemmatizer)

                config["tf-idf"]["preprocessor"] = preprocessor

            else:
                raise KeyError(
                    f"Unknown lemmatizer {lemmatization}. Available lemmatizers: none, pymorphy2."
                )

    # logreg
    if ("logreg" not in config) or (config["logreg"] is None):
        config["logreg"] = {}

    return config
