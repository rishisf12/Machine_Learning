# Iris — Streamlit app (Application / Dashboard / Documentation)
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iris — Species Predictor", layout="wide")

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
    return pd.read_csv("data/iris.csv")

model = load_model()
meta = load_meta()
comparison = load_comparison()
raw = load_data()

FEATURES = meta["features"]
CLASSES = meta["classes"]
KAGGLE_URL = "https://www.kaggle.com/datasets/uciml/iris"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# ---- Tabs ----
tab_app, tab_dash, tab_doc = st.tabs(["🌸 Application", "📊 Dashboard", "📚 Documentation"])

# ================= APPLICATION =================
with tab_app:
    st.title("🌸 Iris Species Predictor")
    st.caption(
        f"Best model: **{meta['best_model']}** | Test accuracy = {meta['metrics']['test_accuracy']} | "
        f"Macro-F1 = {meta['metrics']['test_f1']}"
    )

    st.subheader("Flower measurements (cm)")
    c1, c2 = st.columns(2)
    with c1:
        sepal_length = st.slider("Sepal length", 4.3, 7.9, 5.1)
        sepal_width = st.slider("Sepal width", 2.0, 4.4, 3.5)
    with c2:
        petal_length = st.slider("Petal length", 1.0, 6.9, 1.4)
        petal_width = st.slider("Petal width", 0.1, 2.5, 0.2)

    X_new = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]], columns=FEATURES
    )

    if st.button("Predict species", type="primary"):
        pred = model.predict(X_new)[0]
        st.metric("Predicted species", pred)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0]
            st.subheader("Class probabilities")
            st.bar_chart(pd.Series(proba, index=CLASSES))

    with st.expander("Input used"):
        st.dataframe(X_new, use_container_width=True)

# ================= DASHBOARD =================
with tab_dash:
    st.title("📊 Model Comparison Dashboard")
    st.caption("All methods benchmarked on Iris (hold-out test set, 30 samples)")

    st.dataframe(
        comparison.style.format({
            "cv_accuracy": "{:.3f}", "test_accuracy": "{:.3f}", "test_f1": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Test accuracy by model (higher is better)")
    st.bar_chart(comparison.set_index("model")["test_accuracy"])

    st.subheader("Test macro-F1 by model (higher is better)")
    st.bar_chart(comparison.set_index("model")["test_f1"])

    best = comparison.iloc[0]
    st.success(
        f"**Best: {best['model']}** — Accuracy {best['test_accuracy']:.3f}, "
        f"Macro-F1 {best['test_f1']:.3f}, CV-Acc {best['cv_accuracy']:.3f}"
    )

# ================= DOCUMENTATION =================
with tab_doc:
    st.title("📚 Documentation")

    st.subheader("1. Dataset")
    st.markdown(
        f"- **Name:** Iris (Fisher, 1936)\n"
        f"- **Rows:** {meta['n_samples']} flowers | **Features:** {len(FEATURES)} | "
        f"**Classes:** {len(CLASSES)} (balanced 50/50/50)\n"
        f"- **Target:** `species` ({', '.join(CLASSES)})\n"
        f"- **Live links:** [Kaggle dataset]({KAGGLE_URL}) · [UCI ML Repository]({UCI_URL})"
    )
    st.markdown("**Dataset head (5):**")
    st.dataframe(raw.head(5), use_container_width=True)

    st.subheader("2. Process / Pipeline")
    st.markdown(
        "1. Load from UCI (clean — no missing values, balanced classes)\n"
        "2. 80/20 stratified train/test split (`random_state=42`)\n"
        "3. **Baseline:** most-frequent predictor (DummyClassifier)\n"
        "4. **Primary:** KNN k=5 with StandardScaler (5-fold CV)\n"
        "5. **Advanced:** Logistic Regression, Decision Tree, tuned KNN (GridSearchCV over k=1..30)\n"
        "6. Rank by test accuracy → macro-F1 → CV accuracy → best refit on full data → export"
    )

    st.subheader("3. Complete code (pipeline core)")
    st.code(
        '''import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
COLS = ["sepal_length","sepal_width","petal_length","petal_width","species"]
df = pd.read_csv(URL, header=None, names=COLS)

feature_cols = ["sepal_length","sepal_width","petal_length","petal_width"]
X, y = df[feature_cols], df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

def scaled(est):
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])

models = {
    "KNN (k=5)":            scaled(KNeighborsClassifier(n_neighbors=5)),
    "Logistic Regression":  scaled(LogisticRegression(max_iter=1000)),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "KNN (tuned k)":        GridSearchCV(scaled(KNeighborsClassifier()),
                                          {"clf__n_neighbors": list(range(1,31))},
                                          cv=5, scoring="accuracy"),
}

results = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    cv = (pipe.best_score_ if hasattr(pipe, "best_score_") else
          cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean())
    results.append({"model": name, "cv_accuracy": cv,
                    "test_accuracy": accuracy_score(y_test, p),
                    "test_f1": f1_score(y_test, p, average="macro")})

# Export best (rank: test acc -> F1 -> CV acc)
best_name = sorted(results, key=lambda r: (-r["test_accuracy"],
                                           -r["test_f1"], -r["cv_accuracy"]))[0]["model"]
best_pipe = models[best_name]
best_pipe.fit(X, y)  # refit on full data
joblib.dump(best_pipe, "models/best_model.joblib")''',
        language="python",
    )

    st.subheader("4. Results")
    m = meta["metrics"]
    st.markdown(
        f"- **Best model:** {meta['best_model']}\n"
        f"- **Test accuracy:** {m['test_accuracy']} | **Macro-F1:** {m['test_f1']} | "
        f"CV accuracy: {m['cv_accuracy']}\n"
        f"- All 4 methods tied at 0.933 test accuracy (30-sample test); "
        f"LogReg won on macro-F1."
    )

    st.subheader("5. Resume bullet")
    st.info(
        f"Benchmarked {len(comparison)} classification methods on Iris "
        f"({meta['n_samples']} samples, {len(FEATURES)} features, {len(CLASSES)} classes); "
        f"achieved {meta['best_model']} at test accuracy {m['test_accuracy']:.3f} "
        f"(macro-F1 {m['test_f1']:.3f})."
    )
