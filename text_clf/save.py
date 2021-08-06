import json
import shutil
from typing import Any, Dict

import joblib
from sklearn.pipeline import Pipeline


def save_model(
    pipe: Pipeline,
    target_names_mapping: Dict[int, str],
    config: Dict[str, Any],
) -> None:
    """
    Save model pipeline (tf-idf + model), target names mapping and config.

    :param Pipeline pipe: model pipeline (tf-idf + model).
    :param Dict[int, str] target_names_mapping: name for each class.
    :param Dict[str, Any] config: config.
    :return:
    """

    # save pipe
    joblib.dump(pipe, config["path_to_save_model"])

    # save target names mapping
    with open(config["path_to_save_target_names_mapping"], mode="w") as fp:
        json.dump(target_names_mapping, fp)

    # save config
    shutil.copy2(config["path_to_config"], config["path_to_save_folder"])
