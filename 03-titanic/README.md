# Titanic — Survival Classification Benchmark

Predict passenger survival on the Titanic. **Stage A / Dataset #3** of the ML resume curriculum.

- **Dataset:** [Kaggle competition](https://www.kaggle.com/competitions/titanic) · [public mirror](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) — 891 passengers, 8 model features, target `Survived`
- **Best model:** Logistic Regression — test **accuracy 0.838, AUC 0.876, F1 0.782** (+0.223 vs baseline)

## Pipeline

1. Load `train.csv` (891 rows) from public mirror (Age 177 / Cabin 687 / Embarked 2 missing)
2. Feature engineering: Title from Name (rare-title grouping), FamilySize, IsAlone; drop PassengerId/Name/Ticket/Cabin
3. 80/20 stratified split (`random_state=42`)
4. Preprocessing: numeric → median-impute + scale; categorical → mode-impute + one-hot (ColumnTransformer)
5. Baseline: most-frequent → Primary: LogReg → Advanced: Decision Tree, Random Forest
6. Rank by test accuracy → AUC → CV accuracy → refit best on full data → export

## Results

| Rank | Method | CV-Acc | Test Acc | F1 | AUC |
|---|---|---|---|---|---|
| 1 | **Logistic Regression** | 0.819 | **0.838** | **0.782** | **0.876** |
| 2 | Random Forest | 0.817 | 0.821 | 0.742 | 0.854 |
| 3 | Decision Tree | 0.803 | 0.793 | 0.689 | 0.832 |
| — | Baseline (most-frequent) | — | 0.615 | 0.000 | 0.500 |

## Files

```
03-titanic/
├── 03-titanic.ipynb         # executed notebook (EDA → feature eng → models → export)
├── build_notebook.py        # builds + executes the notebook
├── app.py                   # Streamlit app: Application / Dashboard / Documentation
├── README.md                # this file
├── requirements.txt         # for Streamlit Community Cloud
├── data/titanic.csv         # raw dataset copy
└── models/
    ├── best_model.joblib    # deployed ColumnTransformer+LogReg pipeline
    ├── model_metadata.json  # features / metrics
    └── comparison.csv       # all-model results
```

## Run the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy free on [Streamlit Community Cloud](https://share.streamlit.io) — point at `app.py`.

## Resume bullet

> Benchmarked 3 classification methods on Titanic (891 passengers, 8 features); achieved Logistic Regression at test accuracy 0.838 (AUC 0.876, F1 0.782) +0.223 accuracy vs baseline.
