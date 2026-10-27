# Builds and executes 03-titanic.ipynb (baseline -> primary -> advanced -> comparison -> export)
import nbformat as nbf

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# Titanic — Survival Classification Benchmark (Dataset #3)

**Domain:** Social | **Task:** Predict passenger survival

Workflow: **baseline → primary → one advanced → single comparison table → export → resume bullet.**

- **Baseline:** most-frequent predictor (DummyClassifier)
- **Primary:** Logistic Regression
- **Advanced:** Decision Tree, Random Forest
- **Metrics:** Accuracy, ROC-AUC, F1 (5-fold CV on train, stratified hold-out test)

> Data: Titanic `train.csv` (891 rows). Loaded from a public mirror identical to the
> [Kaggle competition](https://www.kaggle.com/competitions/titanic) file (no API key needed).""")

code("""import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)""")

code("""# Load Titanic (public mirror == Kaggle train.csv)
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(URL)

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/titanic.csv", index=False)  # save raw copy for the Streamlit app

print(f"Rows: {len(df)}, Cols: {df.shape[1]}")
df.head()""")

md("""## 1. EDA & Feature Engineering""")

code("""# Missing values: Age ~20%, Cabin ~77%, Embarked 2
print("Missing per column:")
print(df.isna().sum()[lambda s: s > 0])
print()
print("Survival balance:")
print(df["Survived"].value_counts(normalize=True).round(3))""")

code("""# Feature engineering: Title from Name, FamilySize, IsAlone
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\\.", expand=False)
df["Title"] = df["Title"].replace(
    ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona","Mlle","Ms","Mme"],
    "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

print("Title distribution:")
print(df["Title"].value_counts())""")

code("""# Model features (drop PassengerId, Name, Ticket, Cabin — high-cardinality / redundant)
NUMERIC = ["Age", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Survived"

df = df[NUMERIC + CATEGORICAL + [TARGET]]
print("Features:", FEATURES)""")

md("""## 2. Train / Test Split + Baseline""")

code("""X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")""")

code("""# Preprocessor: impute + scale numerics, impute + one-hot categoricals
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
])

# Baseline: always predict the most frequent class
baseline = Pipeline([("prep", preprocess),
                     ("clf", DummyClassifier(strategy="most_frequent"))]).fit(X_train, y_train)
y_base = baseline.predict(X_test)
y_base_p = baseline.predict_proba(X_test)[:, 1]
print(f"Baseline  Acc={accuracy_score(y_test, y_base):.3f}  "
      f"F1={f1_score(y_test, y_base):.3f}  AUC={roc_auc_score(y_test, y_base_p):.3f}")
baseline_acc = accuracy_score(y_test, y_base)""")

md("""## 3. Primary — Logistic Regression""")

code("""cv_rows = []
def evaluate(name, pipe):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    cv_acc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()
    row = {
        "model": name,
        "cv_accuracy": cv_acc,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_auc": roc_auc_score(y_test, y_prob),
    }
    cv_rows.append(row)
    print(f"{name:22s} CV-Acc={cv_acc:.3f}  Acc={row['test_accuracy']:.3f}  "
          f"F1={row['test_f1']:.3f}  AUC={row['test_auc']:.3f}")

logreg = Pipeline([("prep", preprocess),
                   ("clf", LogisticRegression(max_iter=1000))])
evaluate("Logistic Regression", logreg)""")

md("""## 4. Advanced — Decision Tree, Random Forest""")

code("""tree = Pipeline([("prep", preprocess),
                 ("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))])
evaluate("Decision Tree", tree)

rf = Pipeline([("prep", preprocess),
               ("clf", RandomForestClassifier(n_estimators=200, max_depth=6,
                                              random_state=RANDOM_STATE))])
evaluate("Random Forest", rf)""")

md("""## 5. Comparison Table (all methods)""")

code("""results = pd.DataFrame(cv_rows).sort_values(
    ["test_accuracy", "test_auc", "cv_accuracy"], ascending=False).reset_index(drop=True)
results.insert(0, "rank", results.index + 1)
results.round(3)""")

code("""best = results.iloc[0]
print("BEST MODEL:", best["model"])
print(f"  CV-Acc={best['cv_accuracy']:.3f} | Acc={best['test_accuracy']:.3f} | "
      f"F1={best['test_f1']:.3f} | AUC={best['test_auc']:.3f}")""")

md("""## 6. Export Best Model (for Streamlit deployment)""")

code("""import joblib, json

# Refit the winning pipeline on ALL data for deployment
fitted_lookup = {"Logistic Regression": logreg, "Decision Tree": tree, "Random Forest": rf}
best_obj = fitted_lookup[best["model"]]
best_obj.fit(X, y)  # refit on full dataset (metrics above stay hold-out based)

os.makedirs("models", exist_ok=True)
joblib.dump(best_obj, "models/best_model.joblib")
results.to_csv("models/comparison.csv", index=False)

meta = {
    "dataset": "Titanic",
    "target": "Survived",
    "features": FEATURES,
    "numeric": NUMERIC,
    "categorical": CATEGORICAL,
    "best_model": best["model"],
    "metrics": {
        "cv_accuracy": round(float(best["cv_accuracy"]), 3),
        "test_accuracy": round(float(best["test_accuracy"]), 3),
        "test_f1": round(float(best["test_f1"]), 3),
        "test_auc": round(float(best["test_auc"]), 3),
    },
    "n_samples": int(len(df)),
    "loader": "joblib.load('models/best_model.joblib')",
}
with open("models/model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved models/best_model.joblib + models/comparison.csv + models/model_metadata.json")
print(json.dumps(meta, indent=2))""")

md("""## 7. Resume Bullet""")

code("""bullet = (
    f"Benchmarked {len(results)} classification methods on Titanic ({len(df)} passengers, "
    f"{len(FEATURES)} features); achieved {best['model']} at test accuracy "
    f"{best['test_accuracy']:.3f} (AUC {best['test_auc']:.3f}, F1 {best['test_f1']:.3f}) "
    f"+{best['test_accuracy'] - baseline_acc:.3f} accuracy vs baseline."
)
print("Resume bullet (Titanic):")
print("\\u2022 " + bullet)""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.10"}

import nbclient
client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

nbf.write(nb, "03-titanic.ipynb")
print("Saved executed notebook: 03-titanic.ipynb")
