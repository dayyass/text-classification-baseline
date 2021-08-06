from typing import Any, Dict, Tuple

import pandas as pd


def load_data(
    config: Dict[str, Any]
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load data.

    :param Dict[str, Any] config: config.
    :return: X_train, X_valid, y_train, y_valid.
    :rtype: Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
    """

    text_column = config["data"]["text_column"]
    target_column = config["data"]["target_column"]

    sep = config["data"]["sep"]
    usecols = [text_column, target_column]

    df_train = pd.read_csv(
        config["data"]["train_data_path"],
        sep=sep,
        usecols=usecols,
    )

    df_valid = pd.read_csv(
        config["data"]["valid_data_path"],
        sep=sep,
        usecols=usecols,
    )

    X_train = df_train[text_column]
    X_valid = df_valid[text_column]
    y_train = df_train[target_column]
    y_valid = df_valid[target_column]

    return X_train, X_valid, y_train, y_valid
