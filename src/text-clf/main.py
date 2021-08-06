import ast
import datetime
import json
import random
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def get_argparse() -> ArgumentParser:
    """
    Get argument parser.

    :return: argument parser.
    :rtype: ArgumentParser
    """

    parser = ArgumentParser(prog="text-clf")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="config.yaml",
        help="Path to config",
    )

    return parser


def get_config(path_to_config: str) -> Dict[str, Any]:
    """
    Get config.

    :param str path_to_config: path to config.
    :return: config.
    :rtype: Dict[str, Any]
    """

    with open(path_to_config, mode="r") as fp:
        config = yaml.safe_load(fp)

    return config


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.

    :param int seed: seed.
    """

    random.seed(seed)
    np.random.seed(seed)


def load_data(
    config: Dict[str, Any]
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load data.

    :param Dict[str, Any] config: config.
    :return: X_train, X_test, y_train, y_test.
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


def save_model(
    pipe: Pipeline,
    target_names_mapping: Dict[int, str],
    config: Dict[str, Any],
    path_to_config: str,
) -> None:
    """
    Save model pipeline (tf-idf + model), target names mapping and config.

    :param Pipeline pipe: model pipeline (tf-idf + model).
    :param Dict[int, str] target_names_mapping: name for each class.
    :param Dict[str, Any] config: config.
    :param str path_to_config: path to config.
    :return:
    """

    now = datetime.datetime.now()
    filename = f"model_{now.date()}_{now.time()}"
    path_to_save_folder = Path(config["path_to_save_folder"]) / filename

    # make dirs if not exist
    path_to_save_folder.absolute().mkdir(parents=True, exist_ok=True)

    path_to_save_model = path_to_save_folder / "model.joblib"
    path_to_save_target_names_mapping = path_to_save_folder / "target_names.json"

    # save pipe
    joblib.dump(pipe, path_to_save_model)

    # save target names mapping
    with open(path_to_save_target_names_mapping, mode="w") as fp:
        json.dump(target_names_mapping, fp)

    # save config
    shutil.copy2(path_to_config, path_to_save_folder)


def main() -> int:
    """
    Main function to train baseline model.

    :return: exit code.
    :rtype: int
    """

    # argument parser
    parser = get_argparse()
    args = parser.parse_args()

    # load config
    config = get_config(args.config)

    # reproducibility
    set_seed(config["seed"])

    # load data
    print("Loading data...")

    X_train, X_test, y_train, y_test = load_data(config)

    print(f"Train dataset size: {X_train.shape[0]}")
    print(f"Test dataset size: {X_test.shape[0]}")

    # label encoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    target_names = [str(cls) for cls in le.classes_.tolist()]
    target_names_mapping = {i: cls for i, cls in enumerate(target_names)}

    # tf-idf
    if ("tf-idf" not in config) or (config["tf-idf"] is None):
        config["tf-idf"] = {}
    if "ngram_range" in config["tf-idf"]:
        config["tf-idf"]["ngram_range"] = ast.literal_eval(
            config["tf-idf"]["ngram_range"]
        )

    vectorizer = TfidfVectorizer(**config["tf-idf"])

    # logreg
    if ("logreg" not in config) or (config["logreg"] is None):
        config["logreg"] = {}

    clf = LogisticRegression(
        **config["logreg"],
        random_state=config["seed"],
    )

    # pipeline
    print("\nFitting TF-IDF + LogReg model...")

    pipe = Pipeline(
        [
            ("tf-idf", vectorizer),
            ("logreg", clf),
        ],
        verbose=config["verbose"],
    )

    start_time = time.time()
    pipe.fit(X_train, y_train)

    print(f"Fitting time: {(time.time() - start_time):.2f} seconds")

    print(f"\nTF-IDF number of features: {len(pipe['tf-idf'].vocabulary_)}")

    # metrics
    print("\nCalculating metrics...")

    print("Train classification report:")

    y_pred_train = pipe.predict(X_train)
    print(
        classification_report(
            y_true=y_train,
            y_pred=y_pred_train,
            target_names=target_names,
        )
    )

    print("Test classification report:")

    y_pred_test = pipe.predict(X_test)
    print(
        classification_report(
            y_true=y_test,
            y_pred=y_pred_test,
            target_names=target_names,
        )
    )

    # save model
    print("Saving the model...")

    save_model(
        pipe=pipe,
        target_names_mapping=target_names_mapping,
        config=config,
        path_to_config=args.config,
    )

    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
