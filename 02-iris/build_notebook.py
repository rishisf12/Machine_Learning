# Builds and executes 02-iris.ipynb (baseline -> primary -> advanced -> comparison -> export)
import nbformat as nbf

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# Iris — Classification Benchmark (Dataset #2)

**Domain:** Biology | **Task:** Classify iris species from petal/sepal measurements

Workflow: **baseline → primary → one advanced → single comparison table → export → resume bullet.**

- **Baseline:** most-frequent predictor (DummyClassifier)
- **Primary:** KNN (k=5, scaled)
- **Advanced:** Logistic Regression, Decision Tree, tuned KNN (GridSearchCV over k)
- **Metrics:** Accuracy, macro-F1 (5-fold CV on train, hold-out test)

> Data source: UCI ML Repository `iris` (identical to Kaggle `uciml/iris`).""")

code("""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)""")

code("""# Load Iris (UCI canonical source)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv(URL, header=None, names=COLS)

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/iris.csv", index=False)  # save raw copy for the Streamlit app

print(f"Rows: {len(df)}, Cols: {df.shape[1]}")
df.head()""")

md("""## 1. EDA & Preprocessing""")

code("""# Missing values (Iris is clean)
print("Missing per column:")
print(df.isna().sum())""")

code("""# Class balance + numeric summary
print("Class balance:")
print(df["species"].value_counts())
print()
df.describe().round(2)""")

md("""## 2. Train / Test Split + Baseline""")

code("""feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[feature_cols]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")""")

code("""# Baseline: always predict the most frequent class
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
y_base = baseline.predict(X_test)
print(f"Baseline  Accuracy={accuracy_score(y_test, y_base):.3f}  "
      f"MacroF1={f1_score(y_test, y_base, average='macro'):.3f}")""")

md("""## 3. Primary — KNN (k=5, scaled)""")

code("""def make_model(name, est, scale=True):
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", est))
    return name, Pipeline(steps)

knn = make_model("KNN (k=5)", KNeighborsClassifier(n_neighbors=5))

cv_rows = []
def evaluate(name, obj, use_best_cv=False):
    obj.fit(X_train, y_train)  # fit first (GridSearchCV needs this before best_score_)
    if use_best_cv:
        cv_acc = obj.best_score_
    else:
        cv_acc = cross_val_score(obj, X_train, y_train, cv=5, scoring="accuracy").mean()
    y_pred = obj.predict(X_test)
    row = {
        "model": name,
        "cv_accuracy": cv_acc,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred, average="macro"),
    }
    cv_rows.append(row)
    print(f"{name:24s} CV-Acc={cv_acc:.3f}  test Acc={row['test_accuracy']:.3f}  F1={row['test_f1']:.3f}")

name, pipe = knn
evaluate(name, pipe)""")

md("""## 4. Advanced — LogReg, Decision Tree, Tuned KNN""")

code("""advanced = [
    make_model("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
]
for name, obj in advanced:
    evaluate(name, obj)""")

code("""# Tuned KNN: GridSearchCV over k (nested 5-fold on the train split)
tuned = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
grid = GridSearchCV(tuned, param_grid={"clf__n_neighbors": list(range(1, 31))},
                    cv=5, scoring="accuracy")
evaluate("KNN (tuned k)", grid, use_best_cv=True)
print("Best k:", grid.best_params_["clf__n_neighbors"])""")

md("""## 5. Comparison Table (all methods)""")

code("""results = pd.DataFrame(cv_rows).sort_values(
    ["test_accuracy", "test_f1", "cv_accuracy"], ascending=False).reset_index(drop=True)
results.insert(0, "rank", results.index + 1)
results.round(3)""")

code("""best = results.iloc[0]
print("BEST MODEL:", best["model"])
print(f"  CV-Acc = {best['cv_accuracy']:.3f} | Test Acc = {best['test_accuracy']:.3f} | "
      f"Test Macro-F1 = {best['test_f1']:.3f}")""")

md("""## 6. Export Best Model (for Streamlit deployment)""")

code("""import joblib, json

# Fitted objects from earlier cells (Pipeline and GridSearchCV both support .fit)
fitted_lookup = {}
for name, obj in [knn, *advanced]:
    fitted_lookup[name] = obj
fitted_lookup["KNN (tuned k)"] = grid

best_obj = fitted_lookup[best["model"]]
best_obj.fit(X, y)  # refit on full dataset (metrics above stay hold-out based)

os.makedirs("models", exist_ok=True)
joblib.dump(best_obj, "models/best_model.joblib")

# Save comparison table for the Streamlit dashboard
results.to_csv("models/comparison.csv", index=False)

meta = {
    "dataset": "Iris",
    "target": "species",
    "classes": sorted(y.unique().tolist()),
    "features": feature_cols,
    "best_model": best["model"],
    "metrics": {
        "cv_accuracy": round(float(best["cv_accuracy"]), 3),
        "test_accuracy": round(float(best["test_accuracy"]), 3),
        "test_f1": round(float(best["test_f1"]), 3),
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
    f"Benchmarked {len(results)} classification methods on Iris ({len(df)} samples, "
    f"{len(feature_cols)} features, 3 classes); achieved {best['model']} at test accuracy "
    f"{best['test_accuracy']:.3f} (macro-F1 {best['test_f1']:.3f})."
)
print("Resume bullet (Iris):")
print("\\u2022 " + bullet)""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.10"}

import nbclient
client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

nbf.write(nb, "02-iris.ipynb")
print("Saved executed notebook: 02-iris.ipynb")
