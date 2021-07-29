import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
df_train = pd.read_csv("data/data_train.csv", index_col=0)
df_test = pd.read_csv("data/data_test.csv", index_col=0)

X_train = df_train["Text"]
X_test = df_test["Text"]
y_train = df_train["Class"]
y_test = df_test["Class"]

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

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
