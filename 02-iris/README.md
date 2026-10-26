# Iris — Classification Benchmark

Classify iris species from sepal/petal measurements. **Stage A / Dataset #2** of the ML resume curriculum.

- **Dataset:** [Iris (Kaggle)](https://www.kaggle.com/datasets/uciml/iris) · [UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) — 150 flowers, 4 features, 3 balanced classes
- **Best model:** Logistic Regression — test **accuracy 0.933, macro-F1 0.933**

## Pipeline

1. Load from UCI (clean — no missing values, 50/50/50 class balance)
2. 80/20 stratified train/test split (`random_state=42`)
3. Baseline: most-frequent predictor → Primary: KNN k=5 (scaled, 5-fold CV) → Advanced: Logistic Regression, Decision Tree, tuned KNN (GridSearchCV over k=1..30)
4. Rank by test accuracy → macro-F1 → CV accuracy → refit best on full data → export `models/best_model.joblib`

## Results

| Rank | Method | CV-Acc | Test Acc | Test Macro-F1 |
|---|---|---|---|---|
| 1 | Logistic Regression | 0.958 | **0.933** | **0.933** |
| 2 | Decision Tree | 0.942 | 0.933 | 0.933 |
| 3 | KNN (k=5) | 0.967 | 0.933 | 0.933 |
| 4 | KNN (tuned k) | 0.967 | 0.933 | 0.933 |
| — | Baseline (most-frequent) | — | 0.333 | 0.167 |

All 4 methods tied at 0.933 test accuracy on the 30-sample test set; LogReg won the tie on macro-F1.

## Files

```
02-iris/
├── 02-iris.ipynb            # executed notebook (EDA → models → comparison → export)
├── build_notebook.py        # builds + executes the notebook
├── app.py                   # Streamlit app: Application / Dashboard / Documentation
├── README.md                # this file
├── requirements.txt         # for Streamlit Community Cloud
├── data/iris.csv            # raw dataset copy
└── models/
    ├── best_model.joblib    # deployed pipeline (joblib.load)
    ├── model_metadata.json  # features / classes / metrics
    └── comparison.csv       # all-model results
```

## Run the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy free on [Streamlit Community Cloud](https://share.streamlit.io) — point at `app.py`.

## Resume bullet

> Benchmarked 4 classification methods on Iris (150 samples, 4 features, 3 classes); achieved Logistic Regression at test accuracy 0.933 (macro-F1 0.933).
