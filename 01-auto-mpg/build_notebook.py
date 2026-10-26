# Builds and executes 01-auto-mpg.ipynb (baseline -> primary -> advanced -> comparison)
import nbformat as nbf

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# Auto MPG — Regression Benchmark (Dataset #1)

**Domain:** Automotive | **Task:** Predict fuel efficiency (MPG) from engine specs

This notebook follows the curriculum workflow:
**baseline → primary → one advanced → single comparison table → resume bullet.**

- **Baseline:** mean predictor (DummyRegressor)
- **Primary:** Linear / Ridge / Lasso regression
- **Advanced:** Polynomial features (deg 2–3) + SVR
- **Metrics:** MAE, RMSE, R² (5-fold CV on train, final hold-out test)

> Data source: UCI Machine Learning Repository `auto-mpg` (identical content to the
> Kaggle `uciml/autompg-dataset`). Loaded from the canonical UCI URL for reproducibility
> without Kaggle API keys.""")

code("""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)""")

code("""# Load Auto MPG (UCI canonical source; '?' marks missing horsepower)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
COLS = ["mpg","cylinders","displacement","horsepower","weight",
        "acceleration","model_year","origin","car_name"]
df = pd.read_csv(URL, sep=r"\\s+", header=None, names=COLS, na_values="?")

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/auto-mpg.csv", index=False)  # save raw copy for the Streamlit app

print(f"Rows: {len(df)}, Cols: {df.shape[1]}")
df.head()""")

md("""## 1. EDA & Preprocessing""")

code("""# Missing values: only 'horsepower' has holes
print("Missing per column:")
print(df.isna().sum()[lambda s: s > 0])""")

code("""# Numeric summary of the target and features
df[["mpg","cylinders","displacement","horsepower","weight","acceleration"]].describe().round(2)""")

code("""# Impute missing horsepower with the median, drop the car-name string column
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
df = df.drop(columns=["car_name"])
df.head()""")

md("""## 2. Train / Test Split + Baseline""")

code("""feature_cols = ["cylinders","displacement","horsepower","weight",
                 "acceleration","model_year","origin"]
X = df[feature_cols]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")""")

code("""# Baseline: predict the training-set mean (DummyRegressor)
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
y_base = baseline.predict(X_test)
print(f"Baseline  MAE={mean_absolute_error(y_test, y_base):.3f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_test, y_base)):.3f}  "
      f"R2={r2_score(y_test, y_base):.3f}")""")

md("""## 3. Primary Methods — Linear / Ridge / Lasso""")

code("""def make_model(name, est, scale=False):
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", est))
    return name, Pipeline(steps)

primary = [
    make_model("LinearRegression", LinearRegression()),
    make_model("Ridge (alpha=1.0)", Ridge(alpha=1.0), scale=True),
    make_model("Lasso (alpha=0.01)", Lasso(alpha=0.01, max_iter=10000), scale=True),
]

cv_rows = []
for name, pipe in primary:
    cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=5,
                               scoring="neg_root_mean_squared_error").mean()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    cv_rows.append({
        "model": name,
        "cv_rmse": cv_rmse,
        "test_mae": mean_absolute_error(y_test, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_r2": r2_score(y_test, y_pred),
    })
    print(f"{name:22s} CV-RMSE={cv_rmse:.3f}  test RMSE={cv_rows[-1]['test_rmse']:.3f}  R2={cv_rows[-1]['test_r2']:.3f}")""")

md("""## 4. Advanced — Polynomial Features + SVR""")

code("""advanced = [
    make_model("Poly(deg2)+Ridge",
               Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                         ("ridge", Ridge(alpha=1.0))]), scale=True),
    make_model("Poly(deg3)+Ridge",
               Pipeline([("poly", PolynomialFeatures(degree=3, include_bias=False)),
                         ("ridge", Ridge(alpha=1.0))]), scale=True),
    make_model("SVR (RBF)",
               SVR(kernel="rbf", C=10.0, epsilon=0.1), scale=True),
]

for name, pipe in advanced:
    cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=5,
                               scoring="neg_root_mean_squared_error").mean()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    cv_rows.append({
        "model": name,
        "cv_rmse": cv_rmse,
        "test_mae": mean_absolute_error(y_test, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_r2": r2_score(y_test, y_pred),
    })
    print(f"{name:22s} CV-RMSE={cv_rmse:.3f}  test RMSE={cv_rows[-1]['test_rmse']:.3f}  R2={cv_rows[-1]['test_r2']:.3f}")""")

md("""## 5. Comparison Table (all methods)""")

code("""results = pd.DataFrame(cv_rows).sort_values("test_rmse").reset_index(drop=True)
results.insert(0, "rank", results.index + 1)
results.round(3)

# Save comparison table for the Streamlit dashboard
import os
os.makedirs("models", exist_ok=True)
results.to_csv("models/comparison.csv", index=False)
results.round(3)""")

code("""best = results.iloc[0]
print("BEST MODEL:", best["model"])
print(f"  CV-RMSE = {best['cv_rmse']:.3f} | Test MAE = {best['test_mae']:.3f} | "
      f"Test RMSE = {best['test_rmse']:.3f} | Test R2 = {best['test_r2']:.3f}")""")

md("""## 6. Export Best Model (for Streamlit deployment)""")

code("""import joblib, json, os

# Rebuild the winning pipeline definition and refit on ALL data for deployment
all_pipes = primary + advanced
best_pipe = next(p for n, p in all_pipes if n == best["model"])
best_pipe.fit(X, y)  # refit on full dataset (metrics above stay hold-out based)

os.makedirs("models", exist_ok=True)
joblib.dump(best_pipe, "models/best_model.joblib")

meta = {
    "dataset": "Auto MPG",
    "target": "mpg",
    "features": feature_cols,
    "best_model": best["model"],
    "metrics": {
        "cv_rmse": round(float(best["cv_rmse"]), 3),
        "test_mae": round(float(best["test_mae"]), 3),
        "test_rmse": round(float(best["test_rmse"]), 3),
        "test_r2": round(float(best["test_r2"]), 3),
    },
    "n_samples": int(len(df)),
    "loader": "joblib.load('models/best_model.joblib')",
}
with open("models/model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved models/best_model.joblib + models/model_metadata.json")
print(json.dumps(meta, indent=2))""")

md("""## 7. Resume Bullet""")

code("""delta = best["test_rmse"] - float(results[results['model'] == 'LinearRegression']['test_rmse'].iloc[0])
bullet = (
    f"Benchmarked {len(results)} regression methods on Auto MPG ({len(df)} cars, "
    f"{len(feature_cols)} features); achieved {best['model']} at test RMSE "
    f"{best['test_rmse']:.2f} MPG (R2 {best['test_r2']:.3f}, MAE {best['test_mae']:.2f} mpg) "
    f"- {abs(delta):.2f} RMSE vs baseline linear."
)
print("Resume bullet (Auto MPG):")
print("\\u2022 " + bullet)""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.10"}

import nbclient
client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

nbf.write(nb, "01-auto-mpg.ipynb")
print("Saved executed notebook: 01-auto-mpg.ipynb")
