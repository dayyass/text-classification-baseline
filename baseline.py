import json
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
from sklearn.pipeline import Pipeline
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
print("Loading data...")

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

target_names = [str(cls) for cls in le.classes_.tolist()]
target_names_mapping = {i: cls for i, cls in enumerate(target_names)}


# tf-idf
vectorizer = TfidfVectorizer()


# logreg
clf = LogisticRegression(
    n_jobs=config["n_jobs"],
    random_state=SEED,
)


# pipeline
print("Fitting LogReg + TF-IDF model...")

pipe = Pipeline(
    [
        ("tf-idf", vectorizer),
        ("log-reg", clf),
    ]
)

pipe.fit(X_train, y_train)


# metrics
print("Calculating metrics...")

y_pred = pipe.predict(X_test)
print(
    classification_report(
        y_true=y_test,
        y_pred=y_pred,
        target_names=target_names,
    )
)


# save model
print("Saving the model...")

directory = config["path_to_save_folder"]

if not os.path.exists(directory):
    os.makedirs(directory)

filename_with_ext = config["save_filename"]
path_to_save_model = os.path.join(directory, filename_with_ext)

joblib.dump(pipe, path_to_save_model)

filename, _ = os.path.splitext(filename_with_ext)
path_to_save_target_names_mapping = os.path.join(
    directory, f"{filename}_target_names.json"
)

with open(path_to_save_target_names_mapping, mode="w") as fp:
    json.dump(target_names_mapping, fp)

print("Done!")
