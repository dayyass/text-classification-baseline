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


train_bunch = fetch_20newsgroups(subset="train")
test_bunch = fetch_20newsgroups(subset="test")

df_train = make_df_from_bunch(train_bunch)
df_test = make_df_from_bunch(test_bunch)

if os.getcwd().endswith(r"/text_classification_baseline/data"):
    path_to_save = os.getcwd()
elif os.getcwd().endswith(r"/text_classification_baseline"):
    path_to_save = os.path.join(os.getcwd(), "data")
else:
    raise Exception(
        "Run script from `text_classification_baseline` folder or `text_classification_baseline/data` folder."
    )

df_train.to_csv(os.path.join(path_to_save, "train.csv"), index=False)
df_test.to_csv(os.path.join(path_to_save, "test.csv"), index=False)
