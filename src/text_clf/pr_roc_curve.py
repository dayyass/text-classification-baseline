import json
import os
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from .config import get_config
from .data import load_data

__all__ = [
    "get_precision_recall_curve",
    "get_roc_curve",
    "plot_precision_recall_curve",
    "plot_roc_curve",
    "plot_precision_recall_f1_curves_for_thresholds",
]


def _get_model_and_data(
    path_to_model_folder: str,
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """Helper function to get model and data.

    Args:
        path_to_model_folder (str): Path to trained model folder.

    Raises:
        Exception: Not a binary classification error.
        FileNotFoundError: No config error.
        FileNotFoundError: More then 1 config error.

    Returns:
        Tuple[Pipeline, np.ndarray, np.ndarray]: model, X_test, y_test.
    """

    # path_to_model
    path_to_model = os.path.join(path_to_model_folder, "model.joblib")

    # path_to_target_names
    path_to_target_names = os.path.join(path_to_model_folder, "target_names.json")
    with open(path_to_target_names, mode="r") as fp:
        target_names = json.load(fp)
    assert (
        len(target_names) == 2
    ), f"The model must have 2 classes, but has {len(target_names)} classes."

    # path_to_config
    path_to_model_folder_yaml_list = [
        file for file in os.listdir(path_to_model_folder) if file.endswith(".yaml")
    ]
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
    """Get precision and recall metrics for precision-recall curve.

    Args:
        path_to_model_folder (str): Path to trained model folder.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: precision, recall, thresholds.
    """

    model, X_test, y_test = _get_model_and_data(path_to_model_folder)

    y_test_probas_pred = model.predict_proba(X_test)

    assert (
        y_test_probas_pred.shape[-1] == 2
    ), f"The model must have 2 classes, but has {y_test_probas_pred.shape[-1]} classes."

    precision, recall, thresholds = precision_recall_curve(
        y_true=y_test, probas_pred=y_test_probas_pred[:, -1]
    )

    return precision, recall, thresholds


def get_roc_curve(
    path_to_model_folder: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get fpr and tpr metrics for roc curve.

    Args:
        path_to_model_folder (str): Path to trained model folder.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: fpr, tpr, thresholds
    """

    model, X_test, y_test = _get_model_and_data(path_to_model_folder)

    y_test_probas_pred = model.predict_proba(X_test)

    assert (
        y_test_probas_pred.shape[-1] == 2
    ), f"The model must have 2 classes, but has {y_test_probas_pred.shape[-1]} classes."

    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_test_probas_pred[:, -1])

    return fpr, tpr, thresholds


def plot_precision_recall_curve(
    precision: np.ndarray, recall: np.ndarray
) -> PrecisionRecallDisplay:
    """Plot precision-recall curve.

    Args:
        precision (np.ndarray): Precision for different thresholds.
        recall (np.ndarray): Recall for different thresholds.

    Returns:
        PrecisionRecallDisplay: Sklearn display object.
    """

    return PrecisionRecallDisplay(precision=precision, recall=recall)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray) -> RocCurveDisplay:
    """Plot roc curve.

    Args:
        fpr (np.ndarray): False positive rates for different thresholds.
        tpr (np.ndarray): True positive rates for different thresholds.

    Returns:
        RocCurveDisplay: Sklearn display object.
    """

    roc_auc = auc(fpr, tpr)

    return RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)


class PrecisionRecallF1Display:
    """Precision Recall F1 visualization."""

    def __init__(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        thresholds: np.ndarray,
        plot_f1: bool = True,
    ) -> None:
        """Init visualizer.

        Args:
            precision (np.ndarray): Precision for different thresholds.
            recall (np.ndarray): Recall for different thresholds.
            thresholds (np.ndarray): Thresholds.
            plot_f1 (bool): Plot F1 or not. Defaults to True.
        """

        self.precision = precision
        self.recall = recall
        self.thresholds = thresholds

        self.plot_f1 = plot_f1

        if plot_f1:
            self.f1_score = 2 * precision * recall / (precision + recall)

    def plot(self):
        """Plot visualization."""

        _, ax = plt.subplots()

        ax.plot(self.thresholds, self.precision[:-1], label="precision")
        ax.plot(self.thresholds, self.recall[:-1], label="recall")

        if self.plot_f1:
            ax.plot(self.thresholds, self.f1_score[:-1], label="f1-score")

        ax.set(xlabel="Threshold", ylabel="Metrics")
        ax.legend()
        ax.grid()

        self.ax_ = ax
        self.figure_ = ax.figure

        return self


def plot_precision_recall_f1_curves_for_thresholds(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    plot_f1: bool = True,
) -> PrecisionRecallF1Display:
    """Plot precision, recall, f1 curves for thresholds.

    Args:
        precision (np.ndarray): Precision for different thresholds.
        recall (np.ndarray): Recall for different thresholds.
        thresholds (np.ndarray): Thresholds.
        plot_f1 (bool): Plot F1 or not. Defaults to True.

    Returns:
        PrecisionRecallF1Display: Precision Recall F1 visualization.
    """

    return PrecisionRecallF1Display(
        precision=precision, recall=recall, thresholds=thresholds, plot_f1=plot_f1
    )
