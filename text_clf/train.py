import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .data import load_data
from .save import save_model
from .utils import get_config, get_logger, set_seed


def train(path_to_config: str) -> None:
    """
    Function to train baseline model.

    :param str path_to_config: path to config.
    """

    # load config
    config = get_config(path_to_config)

    # get logger
    logger = get_logger(config["path_to_logfile"])

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
    logger.info("\nFitting TF-IDF + LogReg model...")

    pipe = Pipeline(
        [
            ("tf-idf", vectorizer),
            ("logreg", clf),
        ],
        verbose=config["verbose"],
    )

    start_time = time.time()
    pipe.fit(X_train, y_train)

    logger.info(f"Fitting time: {(time.time() - start_time):.2f} seconds")

    logger.info(f"\nTF-IDF number of features: {len(pipe['tf-idf'].vocabulary_)}")

    # metrics
    logger.info("\nCalculating metrics...")

    logger.info("Train classification report:")

    y_pred_train = pipe.predict(X_train)
    logger.info(
        classification_report(
            y_true=y_train,
            y_pred=y_pred_train,
            target_names=target_names,
        )
    )

    logger.info("Valid classification report:")

    y_pred_valid = pipe.predict(X_valid)
    logger.info(
        classification_report(
            y_true=y_valid,
            y_pred=y_pred_valid,
            target_names=target_names,
        )
    )

    # save model
    logger.info("Saving the model...")

    save_model(
        pipe=pipe,
        target_names_mapping=target_names_mapping,
        config=config,
    )

    logger.info("Done!")
