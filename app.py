# src/phase4/app.py

import os
import glob
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ————————————————————————————————————————————————————————————————
# 1) STREAMLIT CONFIG
# ————————————————————————————————————————————————————————————————
st.set_page_config(
    page_title="Airline Passenger Satisfaction Predictor",
    page_icon="✈️",
    layout="wide"
)

st.markdown(
    """
    <style>
      .main-header { font-size:2.5rem; color:#003366; text-align:center; margin-bottom:1rem; }
      .sub-header  { font-size:1.8rem; color:#005599; margin-top:2rem; margin-bottom:1rem; }
      .section     { padding:1.5rem; border-radius:0.5rem; margin-bottom:1rem;
                      background-color:#f8f9fa; border-left:4px solid #003366; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='main-header'>Airline Passenger Satisfaction Predictor</h1>", unsafe_allow_html=True)

# ————————————————————————————————————————————————————————————————
# 2) DYNAMIC PIPELINE LOADER
# ————————————————————————————————————————————————————————————————
@st.cache_resource
def load_pipelines():
    """
    Scan ./models for any *_pipeline.pkl, load them all,
    and return a dict: { 'Logistic Regression': Pipeline, ... }
    """
    pipelines = {}
    model_dir = os.path.join(os.getcwd(), "models")
    for path in glob.glob(os.path.join(model_dir, "*_pipeline.pkl")):
        fname = os.path.basename(path)
        # turn "logistic_regression_pipeline.pkl" → "Logistic Regression"
        model_name = fname.replace("_pipeline.pkl", "").replace("_", " ").title()
        pipelines[model_name] = joblib.load(path)
    return pipelines

pipelines = load_pipelines()
if not pipelines:
    st.error("🚨 No pipelines found in ./models ! Make sure your *_pipeline.pkl files are in src/phase4/models/")
    st.stop()

model_names = list(pipelines.keys())

# ————————————————————————————————————————————————————————————————
# 3) FEATURE LISTS
# ————————————————————————————————————————————————————————————————
numeric_features = [
    "age",
    "flight distance",
    "departure delay in minutes",
    "arrival delay in minutes",
]
categorical_features = [
    "gender",
    "customer type",
    "type of travel",
    "class",
]

all_required = numeric_features + categorical_features

# ————————————————————————————————————————————————————————————————
# 4) PREDICTION HELPERS
# ————————————————————————————————————————————————————————————————
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase col‐names, strip whitespace, and select only required features.
    """
    df = df.copy()
    # lowercase + strip
    df.columns = df.columns.str.lower().str.strip()
    # keep only columns we trained on
    missing = set(all_required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[all_required]

def single_input_df():
    """
    Build a 1-row DataFrame from Streamlit form inputs
    """
    with st.form("single_passenger"):
        st.markdown("<h3>Passenger Details</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            custtype = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
            travel = st.selectbox("Type of Travel", ["Personal Travel", "Business Travel"])
            travelclass = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
        with c2:
            dist = st.number_input("Flight Distance", min_value=0, value=1000)
            ddelay = st.number_input("Departure Delay (min)", min_value=0, value=0)
            adelay = st.number_input("Arrival Delay (min)",   min_value=0, value=0)

        submitted = st.form_submit_button("▶️ Predict")
        if not submitted:
            return None

        data = {
            "age": [age],
            "gender": [gender],
            "customer type": [custtype],
            "type of travel": [travel],
            "class": [travelclass],
            "flight distance": [dist],
            "departure delay in minutes": [ddelay],
            "arrival delay in minutes": [adelay],
        }
        return pd.DataFrame(data)

# ————————————————————————————————————————————————————————————————
# 5) STREAMLIT TABS
# ————————————————————————————————————————————————————————————————
tab1, tab2 = st.tabs(["🔢 Input & Predict", "📊 Model Performance"])

with tab1:
    st.markdown("<h2 class='sub-header'>Data Input & Prediction</h2>", unsafe_allow_html=True)

    input_mode = st.radio("", ["Single Passenger", "Upload CSV"])
    model_choice = st.selectbox("Choose Model", model_names)
    pipeline = pipelines[model_choice]

    if input_mode == "Single Passenger":
        df_in = single_input_df()
        if df_in is None:
            st.info("Fill out the form above and click ▶️ Predict")
            st.stop()
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if not uploaded:
            st.info("Upload a CSV with columns: " + ", ".join(all_required))
            st.stop()
        df_in = pd.read_csv(uploaded)

    # normalize & predict
    try:
        df_proc = normalize_df(df_in)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    # run prediction
    preds = pipeline.predict(df_proc)
    probs = pipeline.predict_proba(df_proc)[:, 1]  # prob of “1”

    df_out = df_in.copy()
    df_out["prediction"] = np.where(preds == 1, "Satisfied", "Dissatisfied")
    df_out["probability"] = np.round(probs, 3)

    # show
    st.success("✅ Prediction complete!")
    st.dataframe(df_out)

    # download
    csv = df_out.to_csv(index=False)
    st.download_button("Download Results", csv, "predictions.csv", "text/csv")

with tab2:
    st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
    st.write(f"You are using **{model_choice}**.")
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        st.write("• This is a **linear** model with coefficients you can inspect.")
    if hasattr(clf, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": clf.feature_names_in_,
            "importance": clf.feature_importances_
        }).sort_values("importance", ascending=False)
        st.write("#### Top features")
        st.table(imp.head(10))

    # st.markdown("*(Any additional static metrics you recorded in Phase 2 can go here.)*")
