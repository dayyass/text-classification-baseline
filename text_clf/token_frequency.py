from collections import Counter
from typing import Counter as CounterType

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .config import get_config
from .data import load_data

__all__ = ["get_token_frequency"]


def get_token_frequency(path_to_config: str) -> CounterType:
    """Get token frequency.

    Args:
        path_to_config (str): Path to config.

    Returns:
        Dict[str, int]: Token frequency.
    """

    # load config
    config = get_config(path_to_config=path_to_config)

    # load data
    X_train, _, _, _ = load_data(config)

    # vectorizer
    vectorizer = CountVectorizer(**config["tf-idf"])
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # token frequency
    counter = np.asarray(X_train_vectorized.sum(axis=0)).squeeze(0)
    token_frequency = Counter(
        {token: counter[idx] for token, idx in vectorizer.vocabulary_.items()}
    )

    return token_frequency
