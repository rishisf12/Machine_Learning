# Builds and executes 04-sms-spam.ipynb (baseline -> primary -> advanced -> comparison -> export)
import nbformat as nbf

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# SMS Spam — Text Classification Benchmark (Dataset #4)

**Domain:** Text | **Task:** Classify SMS as spam or ham

Workflow: **baseline → primary → one advanced → single comparison table → export → resume bullet.**

- **Baseline:** most-frequent predictor (DummyClassifier)
- **Primary:** Multinomial Naive Bayes + TF-IDF
- **Advanced:** Linear SVM + TF-IDF, Logistic Regression + TF-IDF
- **Metrics:** Accuracy, F1, ROC-AUC (5-fold CV on train, stratified hold-out test)

> Data: SMS Spam Collection (5574 rows). Loaded from a public TSV mirror identical to the
> [Kaggle `uciml/sms-spam-collection-dataset`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).""")

code("""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)""")

code("""# Load SMS Spam (public TSV mirror == Kaggle dataset)
URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
df = pd.read_csv(URL, sep="\\t", header=None, names=["label", "message"])
df["label_num"] = (df["label"] == "spam").astype(int)

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/sms-spam.csv", index=False)  # save raw copy for the Streamlit app

print(f"Rows: {len(df)}, Cols: {df.shape[1]}")
df.head()""")

md("""## 1. EDA & Preprocessing""")

code("""# Class imbalance + missing + message length
print("Class balance:")
print(df["label"].value_counts(normalize=True).round(3))
print()
print("Missing:", df.isna().sum().sum())
df["len"] = df["message"].str.len()
df.groupby("label")["len"].describe().round(1)""")

code("""# Split text/label
X = df["message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")""")

code("""# Baseline: always predict the majority class (ham)
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
y_base = baseline.predict(X_test)
print(f"Baseline  Acc={accuracy_score(y_test, y_base):.3f}  "
      f"F1={f1_score(y_test, y_base):.3f}")""")

md("""## 2. Primary — Multinomial Naive Bayes + TF-IDF""")

code("""def make_text_model(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_df=0.9)),
        ("clf", clf),
    ])

def evaluate(name, pipe):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    else:  # LinearSVC has no predict_proba
        y_score = pipe.decision_function(X_test)
    cv_acc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()
    row = {
        "model": name,
        "cv_accuracy": cv_acc,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_auc": roc_auc_score(y_test, y_score),
    }
    cv_rows.append(row)
    print(f"{name:24s} CV-Acc={cv_acc:.3f}  Acc={row['test_accuracy']:.3f}  "
          f"F1={row['test_f1']:.3f}  AUC={row['test_auc']:.3f}")

cv_rows = []
nb = make_text_model(MultinomialNB())
evaluate("Naive Bayes + TF-IDF", nb)""")

md("""## 3. Advanced — Linear SVM, Logistic Regression""")

code("""svm = make_text_model(LinearSVC(max_iter=5000, random_state=RANDOM_STATE))
evaluate("Linear SVM + TF-IDF", svm)

logreg = make_text_model(LogisticRegression(max_iter=2000))
evaluate("LogReg + TF-IDF", logreg)""")

md("""## 4. Comparison Table (all methods)""")

code("""results = pd.DataFrame(cv_rows).sort_values(
    ["test_accuracy", "test_f1", "cv_accuracy"], ascending=False).reset_index(drop=True)
results.insert(0, "rank", results.index + 1)
results.round(3)""")

code("""best = results.iloc[0]
print("BEST MODEL:", best["model"])
print(f"  CV-Acc={best['cv_accuracy']:.3f} | Acc={best['test_accuracy']:.3f} | "
      f"F1={best['test_f1']:.3f} | AUC={best['test_auc']:.3f}")""")

md("""## 5. Export Best Model (for Streamlit deployment)""")

code("""import joblib, json

# Refit the winning pipeline on ALL data for deployment (tfidf is part of the pipeline)
fitted_lookup = {"Naive Bayes + TF-IDF": nb, "Linear SVM + TF-IDF": svm, "LogReg + TF-IDF": logreg}
best_obj = fitted_lookup[best["model"]]
best_obj.fit(X, y)  # refit on full dataset (metrics above stay hold-out based)

os.makedirs("models", exist_ok=True)
joblib.dump(best_obj, "models/best_model.joblib")
results.to_csv("models/comparison.csv", index=False)

meta = {
    "dataset": "SMS Spam",
    "target": "label (spam/ham)",
    "best_model": best["model"],
    "has_proba": bool(hasattr(best_obj, "predict_proba")),
    "metrics": {
        "cv_accuracy": round(float(best["cv_accuracy"]), 3),
        "test_accuracy": round(float(best["test_accuracy"]), 3),
        "test_f1": round(float(best["test_f1"]), 3),
        "test_auc": round(float(best["test_auc"]), 3),
    },
    "n_samples": int(len(df)),
    "input": "raw SMS text",
    "loader": "joblib.load('models/best_model.joblib')",
}
with open("models/model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved models/best_model.joblib + models/comparison.csv + models/model_metadata.json")
print(json.dumps(meta, indent=2))""")

md("""## 6. Resume Bullet""")

code("""bullet = (
    f"Benchmarked {len(results)} text-classification methods on SMS Spam ({len(df)} messages, "
    f"TF-IDF word 1-2 grams); achieved {best['model']} at test accuracy "
    f"{best['test_accuracy']:.3f} (F1 {best['test_f1']:.3f}, AUC {best['test_auc']:.3f}) "
    f"+{best['test_accuracy'] - 0.866:.3f} accuracy vs baseline."
)
print("Resume bullet (SMS Spam):")
print("\\u2022 " + bullet)""")

nb_out = nbf.v4.new_notebook(cells=cells)
nb_out.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb_out.metadata.language_info = {"name": "python", "version": "3.10"}

import nbclient
client = nbclient.NotebookClient(nb_out, timeout=600, kernel_name="python3")
client.execute()

nbf.write(nb_out, "04-sms-spam.ipynb")
print("Saved executed notebook: 04-sms-spam.ipynb")
