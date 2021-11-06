from typing import Any, Dict, Tuple

import pandas as pd


def load_data(
    config: Dict[str, Any]
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load data.

    Args:
        config (Dict[str, Any]): Config.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
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

    df_test = pd.read_csv(
        config["data"]["test_data_path"],
        sep=sep,
        usecols=usecols,
    )

    X_train = df_train[text_column]
    X_test = df_test[text_column]
    y_train = df_train[target_column]
    y_test = df_test[target_column]

    return X_train, X_test, y_train, y_test
