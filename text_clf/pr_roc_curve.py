import glob
import json
import os
from typing import Tuple

import joblib
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from .config import get_config
from .data import load_data


def _get_model_and_data(
    path_to_model_folder: str,
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """TODO"""

    # path_to_model
    path_to_model = os.path.join(path_to_model_folder, "model.joblib")

    # path_to_target_names
    path_to_target_names = os.path.join(path_to_model_folder, "target_names.json")
    with open(path_to_target_names, mode="r") as fp:
        target_names = json.load(fp)
    if len(target_names) != 2:  # not binary classification error
        raise Exception(
            f"The model must have 2 classes, but has {len(target_names)} classes."
        )

    # path_to_config
    path_to_model_folder_yaml_list = glob.glob("*.yaml")
    if len(path_to_model_folder_yaml_list) == 0:  # no config error
        raise FileNotFoundError("There is no config file (with .yaml extension).")
    elif len(path_to_model_folder_yaml_list) > 1:  # more then 1 config error
        raise FileNotFoundError(
            "There are more then one config files (with .yaml extension)."
        )
    path_to_config = os.path.join(
        path_to_model_folder, path_to_model_folder_yaml_list[0]
    )

    # load config
    config = get_config(path_to_config)

    # load data
    _, X_test, _, y_test = load_data(config)

    # load model
    model = joblib.load(path_to_model)

    return model, X_test, y_test


def get_precision_recall_curve(
    path_to_model_folder: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TODO"""

    model, X_test, y_test = _get_model_and_data(path_to_model_folder)

    y_test_probas_pred = model.predict_proba(X_test)
    precision, recall, thresholds = precision_recall_curve(
        y_true=y_test, probas_pred=y_test_probas_pred
    )

    return precision, recall, thresholds


def get_roc_curve(
    path_to_model_folder: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TODO"""

    model, X_test, y_test = _get_model_and_data(path_to_model_folder)

    y_test_probas_pred = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_true=y_test, probas_pred=y_test_probas_pred)

    return fpr, tpr, thresholds


def plot_precision_recall_curve(
    precision: np.np.ndarray, recall: np.np.ndarray
) -> PrecisionRecallDisplay:
    """TODO"""

    return PrecisionRecallDisplay(precision=precision, recall=recall)


def plot_roc_curve(fpr: np.np.ndarray, tpr: np.np.ndarray) -> RocCurveDisplay:
    """TODO"""

    return RocCurveDisplay(fpr=fpr, tpr=tpr)
