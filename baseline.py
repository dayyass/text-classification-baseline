import os
import random
from argparse import ArgumentParser

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# argument parser
parser = ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=False,
    default="config.yaml",
    help="path to config",
)

args = parser.parse_args()


# load config
with open(args.config, mode="r") as fp:
    config = yaml.safe_load(fp)


# reproducibility
SEED = config["seed"]
random.seed(SEED)
np.random.seed(SEED)


# load data
text_column = config["text_column"]
target_column = config["target_column"]

sep = config["sep"]
usecols = [text_column, target_column]

df_train = pd.read_csv(
    config["train_data_path"],
    sep=sep,
    usecols=usecols,
)

df_test = pd.read_csv(
    config["test_data_path"],
    sep=sep,
    usecols=usecols,
)

X_train = df_train[text_column]
X_test = df_test[text_column]
y_train = df_train[target_column]
y_test = df_test[target_column]


# label encoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# tf-idf
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# logreg
clf = LogisticRegression(
    n_jobs=config["n_jobs"],
    random_state=SEED,
)

clf.fit(X_train_tfidf, y_train)


# metrics
y_pred = clf.predict(X_test_tfidf)
print(
    classification_report(
        y_true=y_test,
        y_pred=y_pred,
    )
)


# save model
directory = config["path_to_save_folder"]
filename = config["save_filename"]

if not os.path.exists(directory):
    os.makedirs(directory)

path = os.path.join(directory, filename)
joblib.dump(clf, path)
