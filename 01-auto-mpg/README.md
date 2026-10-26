# Auto MPG — Regression Benchmark

Predict fuel efficiency (MPG) of cars from engine specs. **Stage A / Dataset #1** of the ML resume curriculum.

- **Dataset:** [Auto MPG (Kaggle)](https://www.kaggle.com/datasets/uciml/autompg-dataset) · [UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data) — 398 cars, 7 features, target `mpg`
- **Best model:** SVR (RBF) + StandardScaler — test **RMSE 2.09 MPG, R² 0.919, MAE 1.69**

## Pipeline

1. Load from UCI (6 missing `horsepower` → median imputation, drop `car_name`)
2. 80/20 train/test split (`random_state=42`)
3. Baseline: mean predictor → Primary: Linear / Ridge / Lasso (5-fold CV) → Advanced: Polynomial (deg 2–3) + Ridge, SVR (RBF)
4. Compare on test MAE / RMSE / R² → refit best on full data → export `models/best_model.joblib`

## Results

| Rank | Method | Test RMSE | Test R² |
|---|---|---|---|
| 1 | SVR (RBF) | **2.088** | **0.919** |
| 2 | Poly(deg2)+Ridge | 2.381 | 0.895 |
| 3 | Poly(deg3)+Ridge | 2.412 | 0.892 |
| 4 | LinearRegression | 2.863 | 0.848 |
| 5 | Ridge (α=1.0) | 2.865 | 0.847 |
| 6 | Lasso (α=0.01) | 2.881 | 0.846 |
| — | Baseline (mean) | 7.347 | −0.004 |

## Files

```
01-auto-mpg/
├── 01-auto-mpg.ipynb        # executed notebook (EDA → models → comparison → export)
├── build_notebook.py        # builds + executes the notebook
├── app.py                   # Streamlit app: Application / Dashboard / Documentation
├── data/auto-mpg.csv        # raw dataset copy
└── models/
    ├── best_model.joblib    # deployed pipeline (joblib.load)
    ├── model_metadata.json  # features / target / metrics
    └── comparison.csv       # all-model results
```

## Run the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy free on [Streamlit Community Cloud](https://share.streamlit.io) — point at `app.py`.

## Resume bullet

> Benchmarked 6 regression methods on Auto MPG (398 cars, 7 features); achieved SVR (RBF) at test RMSE 2.09 MPG (R² 0.919, MAE 1.69 mpg) — −0.78 RMSE vs baseline linear.
