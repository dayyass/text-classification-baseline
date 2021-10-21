import logging
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .data import load_data
from .logger import close_logger
from .save import save_model
from .utils import get_grid_search_params, prepare_dict_to_print, set_seed


def _train(
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[Pipeline, Dict[int, str]]:
    """Function to train baseline model.

    Args:
        config (Dict[str, Any]): Config.
        logger (logging.Logger): Logger.

    Returns:
        Tuple[Pipeline, Dict[int, str]]: Model pipeline (tf-idf + logreg) and target names mapping.
    """

    # log config
    with open(config["path_to_config"], mode="r") as fp:
        logger.info(f"Config:\n\n{fp.read()}")

    # reproducibility
    set_seed(config["seed"])

    # load data
    logger.info("Loading data...")

    X_train, X_test, y_train, y_test = load_data(config)

    logger.info(f"Train dataset size: {X_train.shape[0]}")
    logger.info(f"Test dataset size: {X_test.shape[0]}")

    # label encoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    target_names = [str(cls) for cls in le.classes_.tolist()]
    target_names_mapping = {i: cls for i, cls in enumerate(target_names)}

    # tf-idf
    vectorizer = TfidfVectorizer(**config["tf-idf"])

    # logreg
    clf = LogisticRegression(
        **config["logreg"],
        random_state=config["seed"],
    )

    # pipeline
    pipe = Pipeline(
        [
            ("tf-idf", vectorizer),
            ("logreg", clf),
        ],
        verbose=False if config["grid-search"]["do_grid_search"] else True,
    )

    if config["grid-search"]["do_grid_search"]:
        logger.info("Finding best hyper-parameters...")

        grid_search_params = get_grid_search_params(
            config["grid-search"]["grid_search_params_path"]
        )
        grid = GridSearchCV(pipe, **grid_search_params)
        grid.fit(X_train, y_train)

        pipe = grid.best_estimator_

        logger.info(
            f"Best hyper-parameters:\n{prepare_dict_to_print(grid.best_params_)}"
        )

    else:
        logger.info("Fitting TF-IDF + LogReg model...")

        pipe.fit(X_train, y_train)

    logger.info("Done!")
    logger.info(f"TF-IDF number of features: {len(pipe['tf-idf'].vocabulary_)}")

    # metrics
    logger.info("Calculating metrics...")

    y_pred_train = pipe.predict(X_train)
    classification_report_train = classification_report(
        y_true=y_train,
        y_pred=y_pred_train,
        target_names=target_names,
    )
    conf_matrix_train = pd.DataFrame(
        confusion_matrix(
            y_true=y_train,
            y_pred=y_pred_train,
        ),
        columns=target_names,
        index=target_names,
    )

    logger.info(f"Train classification report:\n\n{classification_report_train}")
    logger.info(f"Train confusion matrix:\n\n{conf_matrix_train}\n")

    y_pred_test = pipe.predict(X_test)
    classification_report_test = classification_report(
        y_true=y_test,
        y_pred=y_pred_test,
        target_names=target_names,
    )
    conf_matrix_test = pd.DataFrame(
        confusion_matrix(
            y_true=y_test,
            y_pred=y_pred_test,
        ),
        columns=target_names,
        index=target_names,
    )

    logger.info(f"Test classification report:\n\n{classification_report_test}")
    logger.info(f"Test confusion matrix:\n\n{conf_matrix_test}\n")

    # save model
    logger.info("Saving the model...")

    save_model(
        pipe=pipe,
        target_names_mapping=target_names_mapping,
        config=config,
    )

    logger.info("Done!")

    close_logger(logger)

    return pipe, target_names_mapping
