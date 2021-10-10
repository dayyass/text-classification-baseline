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
    """Save:
        - model pipeline (tf-idf + logreg)
        - target names mapping
        - config
        - hyper-parameters grid (from config)

    Args:
        pipe (Pipeline): Model pipeline (tf-idf + logreg).
        target_names_mapping (Dict[int, str]): Name for each class.
        config (Dict[str, Any]): Config.
    """

    # save pipe
    joblib.dump(pipe, config["path_to_save_model"])

    # save target names mapping
    with open(config["path_to_save_target_names_mapping"], mode="w") as fp:
        json.dump(target_names_mapping, fp)

    # save config
    shutil.copy2(config["path_to_config"], config["path_to_save_folder"])

    # save hyperparams grid
    if config["grid-search"]["do_grid_search"]:
        shutil.copy2(
            config["grid-search"]["grid_search_params_path"],
            config["path_to_save_folder"],
        )
