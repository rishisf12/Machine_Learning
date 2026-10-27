# Titanic — Streamlit app (Application / Dashboard / Documentation)
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic — Survival Predictor", layout="wide")

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
    return pd.read_csv("data/titanic.csv")

model = load_model()
meta = load_meta()
comparison = load_comparison()
raw = load_data()

FEATURES = meta["features"]
KAGGLE_URL = "https://www.kaggle.com/competitions/titanic"
MIRROR_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# ---- Tabs ----
tab_app, tab_dash, tab_doc = st.tabs(["🚢 Application", "📊 Dashboard", "📚 Documentation"])

# ================= APPLICATION =================
with tab_app:
    st.title("🚢 Titanic Survival Predictor")
    st.caption(
        f"Best model: **{meta['best_model']}** | Test accuracy = {meta['metrics']['test_accuracy']} | "
        f"AUC = {meta['metrics']['test_auc']} | F1 = {meta['metrics']['test_f1']}"
    )

    st.subheader("Passenger details")
    c1, c2, c3 = st.columns(3)
    with c1:
        pclass = st.selectbox("Passenger class", [1, 2, 3], index=1)
        sex = st.selectbox("Sex", ["male", "female"])
        embarked = st.selectbox("Embarked", ["S", "C", "Q"])
    with c2:
        age = st.slider("Age", 0.0, 80.0, 28.0)
        fare = st.slider("Fare (£)", 0.0, 512.0, 32.0)
        title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
    with c3:
        sibsp = st.slider("Siblings/Spouses aboard", 0, 8, 0)
        parch = st.slider("Parents/Children aboard", 0, 6, 0)

    family_size = sibsp + parch + 1
    is_alone = int(family_size == 1)
    X_new = pd.DataFrame(
        [[age, fare, family_size, is_alone, pclass, sex, embarked, title]],
        columns=FEATURES,
    )

    if st.button("Predict survival", type="primary"):
        pred = int(model.predict(X_new)[0])
        proba = float(model.predict_proba(X_new)[0][1])
        st.metric("Prediction", "Survived 🟢" if pred == 1 else "Did not survive 🔴",
                  f"P(survive) = {proba:.2f}")
        st.progress(proba)
        if proba >= 0.6:
            st.success("Model leans toward survival.")
        elif proba <= 0.4:
            st.warning("Model leans toward not surviving.")
        else:
            st.info("Borderline case.")

    with st.expander("Engineered input used"):
        st.dataframe(X_new, use_container_width=True)

# ================= DASHBOARD =================
with tab_dash:
    st.title("📊 Model Comparison Dashboard")
    st.caption("All methods benchmarked on Titanic (hold-out test set, 179 passengers)")

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

    st.subheader("Test ROC-AUC by model (higher is better)")
    st.bar_chart(comparison.set_index("model")["test_auc"])

    best = comparison.iloc[0]
    st.success(
        f"**Best: {best['model']}** — Accuracy {best['test_accuracy']:.3f}, "
        f"AUC {best['test_auc']:.3f}, F1 {best['test_f1']:.3f}"
    )

# ================= DOCUMENTATION =================
with tab_doc:
    st.title("📚 Documentation")

    st.subheader("1. Dataset")
    st.markdown(
        f"- **Name:** Titanic — Machine Learning from Disaster\n"
        f"- **Rows:** {meta['n_samples']} passengers | **Model features:** {len(FEATURES)}\n"
        f"- **Target:** `Survived` (0/1, 61.6% / 38.4% split)\n"
        f"- **Missing:** Age 177, Cabin 687 (dropped), Embarked 2 (imputed)\n"
        f"- **Live links:** [Kaggle competition]({KAGGLE_URL}) · [data mirror]({MIRROR_URL})"
    )
    st.markdown("**Dataset head (5):**")
    st.dataframe(raw.head(5), use_container_width=True)

    st.subheader("2. Process / Pipeline")
    st.markdown(
        "1. Load `train.csv` (891 rows) from public mirror\n"
        "2. **Feature engineering:** Title from Name (rare-title grouping), "
        "FamilySize = SibSp+Parch+1, IsAlone; drop PassengerId/Name/Ticket/Cabin\n"
        "3. 80/20 stratified split (`random_state=42`)\n"
        "4. Preprocessing: numeric → median-impute + scale; categorical → mode-impute + one-hot\n"
        "5. **Baseline:** most-frequent → **Primary:** LogReg → **Advanced:** Decision Tree, Random Forest\n"
        "6. Rank by test accuracy → AUC → CV accuracy → refit best on full data → export"
    )

    st.subheader("3. Complete code (pipeline core)")
    st.code(
        '''import re, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(URL)

# Feature engineering
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\\.", expand=False)
df["Title"] = df["Title"].replace(["Lady","Countess","Capt","Col","Don","Dr","Major",
                                   "Rev","Sir","Jonkheer","Dona"], "Rare")
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

NUMERIC = ["Age", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title"]
X, y = df[NUMERIC + CATEGORICAL], df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
])

models = {
    "Logistic Regression": Pipeline([("prep", preprocess),
                                     ("clf", LogisticRegression(max_iter=1000))]),
    "Decision Tree": Pipeline([("prep", preprocess),
                               ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))]),
    "Random Forest": Pipeline([("prep", preprocess),
                               ("clf", RandomForestClassifier(n_estimators=200,
                                                              max_depth=6, random_state=42))]),
}

results = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    p, proba = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
    results.append({"model": name,
                    "cv_accuracy": cross_val_score(pipe, X_train, y_train, cv=5,
                                                   scoring="accuracy").mean(),
                    "test_accuracy": accuracy_score(y_test, p),
                    "test_f1": f1_score(y_test, p),
                    "test_auc": roc_auc_score(y_test, proba)})

# Export best (rank: acc -> AUC -> CV acc)
best_name = sorted(results, key=lambda r: (-r["test_accuracy"], -r["test_auc"],
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
        f"- **Test accuracy:** {m['test_accuracy']} | **AUC:** {m['test_auc']} | "
        f"F1: {m['test_f1']} | CV accuracy: {m['cv_accuracy']}\n"
        f"- Baseline (most-frequent): 0.615 accuracy → **+0.223 lift**"
    )

    st.subheader("5. Resume bullet")
    st.info(
        f"Benchmarked {len(comparison)} classification methods on Titanic "
        f"({meta['n_samples']} passengers, {len(FEATURES)} features); achieved "
        f"{meta['best_model']} at test accuracy {m['test_accuracy']:.3f} "
        f"(AUC {m['test_auc']:.3f}, F1 {m['test_f1']:.3f}) +0.223 accuracy vs baseline."
    )
