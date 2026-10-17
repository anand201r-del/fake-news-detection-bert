"""
OPTIONAL - Classical baselines on the SAME clean data, for your comparison table.
(Logistic Regression, Naive Bayes, Linear SVM with TF-IDF.)  Runs on CPU in minutes.

Run:  python baselines.py        -> reports/baselines.json
"""
import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

DATA = os.path.join("data", "processed")

train = pd.read_csv(os.path.join(DATA, "train.csv")).dropna()
test = pd.read_csv(os.path.join(DATA, "test.csv")).dropna()

vec = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
xtr = vec.fit_transform(train["text"])
xte = vec.transform(test["text"])

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, C=4.0),
    "NaiveBayes": MultinomialNB(alpha=0.1),
    "LinearSVM": LinearSVC(C=0.5),
}

results = {}
for name, clf in models.items():
    clf.fit(xtr, train["label"])
    pred = clf.predict(xte)
    t = test.assign(pred=pred)
    results[name] = {"all": {"accuracy": round(accuracy_score(t.label, t.pred), 4),
                             "f1": round(f1_score(t.label, t.pred), 4)}}
    for kind, g in t.groupby("kind"):
        results[name][kind] = {"accuracy": round(accuracy_score(g.label, g.pred), 4),
                               "f1": round(f1_score(g.label, g.pred), 4)}
    print(name, results[name])

os.makedirs("reports", exist_ok=True)
with open(os.path.join("reports", "baselines.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved reports/baselines.json")
