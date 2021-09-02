import logging
from typing import Any, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .data import load_data
from .save import save_model
from .utils import close_logger, get_grid_search_params, prepare_dict_to_print, set_seed


def _train(
    config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Function to train baseline model.

    :param Dict[str, Any] config: config.
    :param logging.Logger logger: logger.
    """

    # log config
    with open(config["path_to_config"], mode="r") as fp:
        logger.info(f"Config:\n\n{fp.read()}")

    # reproducibility
    set_seed(config["seed"])

    # load data
    logger.info("Loading data...")

    X_train, X_valid, y_train, y_valid = load_data(config)

    logger.info(f"Train dataset size: {X_train.shape[0]}")
    logger.info(f"Valid dataset size: {X_valid.shape[0]}")

    # label encoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_valid = le.transform(y_valid)

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

    logger.info(f"Train classification report:\n\n{classification_report_train}")

    y_pred_valid = pipe.predict(X_valid)
    classification_report_valid = classification_report(
        y_true=y_valid,
        y_pred=y_pred_valid,
        target_names=target_names,
    )

    logger.info(f"Valid classification report:\n\n{classification_report_valid}")

    # save model
    logger.info("Saving the model...")

    save_model(
        pipe=pipe,
        target_names_mapping=target_names_mapping,
        config=config,
    )

    logger.info("Done!")

    close_logger(logger)
