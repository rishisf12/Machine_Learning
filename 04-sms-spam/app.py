# SMS Spam — Streamlit app (Application / Dashboard / Documentation)
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SMS Spam — Classifier", layout="wide")

# ---- Load artifacts ----
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")

@st.cache_data
def load_meta():
    return json.load(open("models/model_metadata.json"))

@st.cache_data
def load_comparison():
    return pd.read_csv("models/comparison.csv")

@st.cache_data
def load_data():
    return pd.read_csv("data/sms-spam.csv")

model = load_model()
meta = load_meta()
comparison = load_comparison()
raw = load_data()

KAGGLE_URL = "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset"
MIRROR_URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
HAS_PROBA = meta.get("has_proba", True)

# ---- Tabs ----
tab_app, tab_dash, tab_doc = st.tabs(["📩 Application", "📊 Dashboard", "📚 Documentation"])

# ================= APPLICATION =================
with tab_app:
    st.title("📩 SMS Spam Classifier")
    st.caption(
        f"Best model: **{meta['best_model']}** | Test accuracy = {meta['metrics']['test_accuracy']} | "
        f"F1 = {meta['metrics']['test_f1']} | AUC = {meta['metrics']['test_auc']}"
    )

    st.subheader("Type an SMS message")
    msg = st.text_area(
        "Message",
        value="Congratulations! You've won a $1000 gift card. Click here to claim now!!!",
        height=120,
    )

    if st.button("Classify", type="primary") and msg.strip():
        pred = int(model.predict([msg])[0])
        label = "SPAM 🚨" if pred == 1 else "HAM ✅"
        if HAS_PROBA:
            proba = float(model.predict_proba([msg])[0][1])
            st.metric("Prediction", label, f"P(spam) = {proba:.2f}")
            st.progress(proba)
        else:
            score = float(model.decision_function([msg])[0])  # LinearSVC: no predict_proba
            st.metric("Prediction", label, f"spam score = {score:+.2f}")
            st.caption("LinearSVM outputs a decision score (positive → spam), not a probability.")
        if pred == 1:
            st.warning("Flagged as spam.")
        else:
            st.success("Looks like a normal message.")

# ================= DASHBOARD =================
with tab_dash:
    st.title("📊 Model Comparison Dashboard")
    st.caption("All methods benchmarked on SMS Spam (hold-out test set, 1115 messages)")

    st.dataframe(
        comparison.style.format({
            "cv_accuracy": "{:.3f}", "test_accuracy": "{:.3f}",
            "test_f1": "{:.3f}", "test_auc": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Test accuracy by model (higher is better)")
    st.bar_chart(comparison.set_index("model")["test_accuracy"])

    st.subheader("Test F1 by model (higher is better)")
    st.bar_chart(comparison.set_index("model")["test_f1"])

    best = comparison.iloc[0]
    st.success(
        f"**Best: {best['model']}** — Accuracy {best['test_accuracy']:.3f}, "
        f"F1 {best['test_f1']:.3f}, AUC {best['test_auc']:.3f}"
    )

# ================= DOCUMENTATION =================
with tab_doc:
    st.title("📚 Documentation")

    st.subheader("1. Dataset")
    st.markdown(
        f"- **Name:** SMS Spam Collection\n"
        f"- **Rows:** {meta['n_samples']} messages | **Features:** TF-IDF (word 1–2 grams, top terms)\n"
        f"- **Target:** `label` (ham 86.6% / spam 13.4% — imbalanced)\n"
        f"- **Missing:** none\n"
        f"- **Live links:** [Kaggle dataset]({KAGGLE_URL}) · [data mirror]({MIRROR_URL})"
    )
    st.markdown("**Dataset head (5):**")
    st.dataframe(raw.head(5), use_container_width=True)

    st.subheader("2. Process / Pipeline")
    st.markdown(
        "1. Load TSV (5572 rows) from public mirror, encode `spam`=1\n"
        "2. 80/20 stratified split (`random_state=42`) — preserves 13.4% spam rate\n"
        "3. **Baseline:** most-frequent (always ham → 0.866 acc, 0 F1)\n"
        "4. Vectorize: `TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.9)`\n"
        "5. **Primary:** Multinomial Naive Bayes → **Advanced:** Linear SVM, Logistic Regression\n"
        "6. Rank by test accuracy → F1 → CV accuracy → refit best on full data → export"
    )

    st.subheader("3. Complete code (pipeline core)")
    st.code(
        '''import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
df = pd.read_csv(URL, sep="\\t", header=None, names=["label", "message"])
df["label_num"] = (df["label"] == "spam").astype(int)

X, y = df["message"], df["label_num"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

def make_text_model(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_df=0.9)),
        ("clf", clf),
    ])

models = {
    "Naive Bayes + TF-IDF": make_text_model(MultinomialNB()),
    "Linear SVM + TF-IDF":  make_text_model(LinearSVC(max_iter=5000, random_state=42)),
    "LogReg + TF-IDF":      make_text_model(LogisticRegression(max_iter=2000)),
}

results = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    score = (pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba")
             else pipe.decision_function(X_test))
    results.append({"model": name,
                    "cv_accuracy": cross_val_score(pipe, X_train, y_train, cv=5,
                                                   scoring="accuracy").mean(),
                    "test_accuracy": accuracy_score(y_test, p),
                    "test_f1": f1_score(y_test, p),
                    "test_auc": roc_auc_score(y_test, score)})

# Export best (rank: acc -> F1 -> CV acc)
best_name = sorted(results, key=lambda r: (-r["test_accuracy"], -r["test_f1"],
                                           -r["cv_accuracy"]))[0]["model"]
best_pipe = models[best_name]
best_pipe.fit(X, y)  # refit on full data
joblib.dump(best_pipe, "models/best_model.joblib")''',
        language="python",
    )

    st.subheader("4. Results")
    m = meta["metrics"]
    st.markdown(
        f"- **Best model:** {meta['best_model']}\n"
        f"- **Test accuracy:** {m['test_accuracy']} | **F1:** {m['test_f1']} | "
        f"AUC: {m['test_auc']} | CV accuracy: {m['cv_accuracy']}\n"
        f"- Baseline (always ham): 0.866 accuracy, 0 F1 → **+0.114 accuracy, +0.921 F1**"
    )

    st.subheader("5. Resume bullet")
    st.info(
        f"Benchmarked {len(comparison)} text-classification methods on SMS Spam "
        f"({meta['n_samples']} messages, TF-IDF word 1-2 grams); achieved "
        f"{meta['best_model']} at test accuracy {m['test_accuracy']:.3f} "
        f"(F1 {m['test_f1']:.3f}, AUC {m['test_auc']:.3f}) +0.114 accuracy vs baseline."
    )
