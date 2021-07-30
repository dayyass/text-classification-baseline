import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from joblib import dump, load

# Load data
df_train = pd.read_csv("data/data_train.csv", usecols=['text', 'target'])
df_test = pd.read_csv("data/data_test.csv", usecols=['text', 'target'])

X_train = df_train["text"]
X_test = df_test["text"]
y_train = df_train["target"]
y_test = df_test["target"]

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
directory = 'saved_models'

if not os.path.exists(directory):
    os.makedirs(directory)

path = f'{directory}/{filename}.joblib'
dump(clf, path)