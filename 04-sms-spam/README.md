# SMS Spam — Text Classification Benchmark

Classify SMS messages as spam or ham. **Stage A / Dataset #4** of the ML resume curriculum.

- **Dataset:** [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) · [public mirror](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv) — 5572 messages, imbalanced (13.4% spam)
- **Best model:** Linear SVM + TF-IDF — test **accuracy 0.980, F1 0.921, AUC 0.989** (+0.114 acc, +0.921 F1 vs baseline)

## Pipeline

1. Load TSV (5572 rows), encode `spam`=1
2. 80/20 stratified split (`random_state=42`)
3. `TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.9)`
4. Baseline: most-frequent → Primary: Multinomial Naive Bayes → Advanced: Linear SVM, Logistic Regression
5. Rank by test accuracy → F1 → CV accuracy → refit best on full data → export

## Results

| Rank | Method | CV-Acc | Test Acc | F1 | AUC |
|---|---|---|---|---|---|
| 1 | **Linear SVM + TF-IDF** | 0.984 | **0.980** | **0.921** | **0.989** |
| 2 | Naive Bayes + TF-IDF | 0.968 | 0.970 | 0.875 | 0.988 |
| 3 | LogReg + TF-IDF | 0.961 | 0.970 | 0.875 | 0.986 |
| — | Baseline (always ham) | — | 0.866 | 0.000 | 0.500 |

## Files

```
04-sms-spam/
├── 04-sms-spam.ipynb        # executed notebook (EDA → models → comparison → export)
├── build_notebook.py        # builds + executes the notebook
├── app.py                   # Streamlit app: Application / Dashboard / Documentation
├── README.md                # this file
├── requirements.txt         # for Streamlit Community Cloud
├── data/sms-spam.csv        # raw dataset copy
└── models/
    ├── best_model.joblib    # deployed TF-IDF + LinearSVM pipeline
    ├── model_metadata.json  # metrics / has_proba flag
    └── comparison.csv       # all-model results
```

## Run the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy free on [Streamlit Community Cloud](https://share.streamlit.io) — point at `app.py`.

## Resume bullet

> Benchmarked 3 text-classification methods on SMS Spam (5572 messages, TF-IDF word 1-2 grams); achieved Linear SVM + TF-IDF at test accuracy 0.980 (F1 0.921, AUC 0.989) +0.114 accuracy vs baseline.
