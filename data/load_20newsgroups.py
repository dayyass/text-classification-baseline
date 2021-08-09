import os

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch


def make_df_from_bunch(bunch: Bunch) -> pd.DataFrame:
    """
    Make pd.DataFrame from 20newsgroups bunch.

    :param Bunch bunch: 20newsgroups bunch.
    :return: 20newsgroups DataFrame.
    :rtype: pd.DataFrame
    """

    df = pd.DataFrame(
        {
            "text": bunch.data,
            "target": bunch.target,
        }
    )
    df["target_name"] = df["target"].map(lambda x: bunch.target_names[x])
    df["target_name_short"] = df["target_name"].map(lambda x: x.split(".")[0])

    return df


def load_20newsgroups() -> None:
    """
    Load 20newsgroups dataset.
    """

    train_bunch = fetch_20newsgroups(subset="train")
    test_bunch = fetch_20newsgroups(subset="test")

    df_train = make_df_from_bunch(train_bunch)
    df_valid = make_df_from_bunch(test_bunch)

    os.makedirs("data", exist_ok=True)

    df_train.to_csv("data/train.csv", index=False)
    df_valid.to_csv("data/valid.csv", index=False)


if __name__ == "__main__":
    load_20newsgroups()
