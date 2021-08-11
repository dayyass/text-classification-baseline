import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .data import load_data
from .save import save_model
from .utils import get_config, get_logger, set_seed


class ParameterOptimizer:

    _params_tfidf = {
        'min_df': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'max_df': [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
    }

    def __init__(self, pipeline, X_train, X_val, y_train, y_val):
        self._pipeline = pipeline
        self._X_train = X_train
        self._X_val = X_val
        self._y_train = y_train
        self._y_val = y_val

    def _get_f1(self):
        self._pipeline.fit(self._X_train, self._y_train)
        _y_pred = self._pipeline.predict(self._X_val)
        _f1_current = f1_score(self._y_val, _y_pred, average='weighted')
        return _f1_current

    def search_opt_params(self):
        _opt_min_df = 0
        _opt_max_df = 1.0
        f1_max = 0

        for min_df in self._params_tfidf['min_df']:
            self._pipeline.set_params(tfidf__min_df=min_df)
            _f1_current = self._get_f1()
            if f1_max < _f1_current:
                f1_max = _f1_current
                _opt_min_df = min_df
            else:
                break

        for max_df in self._params_tfidf['max_df']:
            self._pipeline.set_params(tfidf__max_df=max_df)
            _f1_current = self._get_f1()
            if f1_max < _f1_current:
                f1_max = _f1_current
                _opt_max_df = max_df
            else:
                break

        return [int(_opt_min_df), float(_opt_max_df)]


def train(path_to_config: str) -> None:
    """
    Function to train baseline model.

    :param str path_to_config: path to config.
    """

    # load config
    config = get_config(path_to_config)

    # mkdir if not exists
    config["path_to_save_folder"].absolute().mkdir(parents=True, exist_ok=True)

    # get logger
    logger = get_logger(config["path_to_save_logfile"])

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
    logger.info("Fitting TF-IDF + LogReg model...")

    pipe = Pipeline(
        [
            ("tfidf", vectorizer),
            ("logreg", clf),
        ],
        verbose=config["verbose"],
    )

    ###### Changes start here
    tf_idf_optimizer = ParameterOptimizer(pipe, X_train, X_valid, y_train, y_valid)
    params = tf_idf_optimizer.search_opt_params()
    opt_min_df, opt_max_df = params[0], params[1]

    ###### Changes end here

    start_time = time.time()
    pipe.set_params(tfidf__min_df=opt_min_df, tfidf__max_df=opt_max_df)
    pipe.fit(X_train, y_train)

    logger.info(f"Fitting time: {(time.time() - start_time):.2f} seconds")

    logger.info(f"TF-IDF number of features: {len(pipe['tfidf'].vocabulary_)}")

    # metrics
    logger.info("Calculating metrics...")

    y_pred_train = pipe.predict(X_train)
    classification_report_train = classification_report(
        y_true=y_train,
        y_pred=y_pred_train,
        target_names=target_names,
    )

    logger.info(f"Train classification report:\n{classification_report_train}")

    y_pred_valid = pipe.predict(X_valid)
    classification_report_valid = classification_report(
        y_true=y_valid,
        y_pred=y_pred_valid,
        target_names=target_names,
    )

    logger.info(f"Valid classification report:\n{classification_report_valid}")

    # save model
    logger.info("Saving the model...")

    save_model(
        pipe=pipe,
        target_names_mapping=target_names_mapping,
        config=config,
    )

    logger.info("Done!")
