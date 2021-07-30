import os
import yaml
import pandas as pd

from argparse import ArgumentParser

from sklearn.preprocessing import LabelEncoder
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
path_to_config = 'config.yaml'
with open(path_to_config, mode="r") as fp:
    config = yaml.safe_load(fp)

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

# Label Encoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Load Model
# clf = load('filename.joblib')

# LogReg
clf = LogisticRegression()
clf.fit(X_train_tfidf, y_train)

# Metrics
y_pred = clf.predict(X_test_tfidf)
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
dump(clf, path)