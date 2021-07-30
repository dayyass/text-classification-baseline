import os
import yaml
import pandas as pd
import numpy as np
import random

from argparse import ArgumentParser

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from joblib import dump, load

# Add ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=False,
    default="config.yaml",
    help="path to config",
)

args = parser.parse_args()

# Load config
with open(args.config, mode="r") as fp:
    config = yaml.safe_load(fp)

# Reproducibility
SEED = config['seed']
random.seed(SEED)
np.random.seed(SEED)

# Load data
text_column = config['text_column']
target_column = config['target_column']

df_train = pd.read_csv(
    config['train_data_path'],
    sep=config['sep'],
    usecols=[
        text_column,
        target_column,
    ]
)

df_test = pd.read_csv(
    config['test_data_path'],
    sep=config['sep'],
    usecols=[
        text_column,
        target_column,
    ]
)


X_train = df_train[text_column]
X_test = df_test[text_column]
y_train = df_train[target_column]
y_test = df_test[target_column]

# Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('log_reg', LogisticRegression(
                    n_jobs=config['n_jobs'],
                    random_state=SEED,
                )
     )
])

model.fit(X_train, y_train)

# Metrics
y_pred = model.predict(X_test)
print(
    classification_report(
        y_true=y_test,
        y_pred=y_pred,
    )
)

# Save model
filename = 'log_reg_tf_idf'
directory = config['path_to_folder']

if not os.path.exists(directory):
    os.makedirs(directory)

path = f'{directory}/{filename}.joblib'
dump(model, path)