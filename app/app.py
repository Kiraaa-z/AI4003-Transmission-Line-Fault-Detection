
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Locate the project folders
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

# Allow imports from src
sys.path.insert(0, str(SRC_DIR))

from logistic import predict


FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


@st.cache_resource
def load_models():
    """Load trained model parameters and the saved RBF SVM."""
    with open(
        RESULTS_DIR / "logistic_set_q.json",
        "r",
        encoding="utf-8",
    ) as file:
        logistic_q = json.load(file)

    rbf_svm = joblib.load(
        MODELS_DIR / "best_rbf_svm.joblib"
    )

    # Calculate Set Q scaling statistics from training data only
    train_df = pd.read_csv(DATA_DIR / "train_raw.csv")
    raw_train = train_df[FEATURES].to_numpy(dtype=float)

    q_train = np.column_stack(
        [raw_train, raw_train ** 2]
    )

    q_mean = q_train.mean(axis=0)
    q_std = q_train.std(axis=0, ddof=1)

    theta_q = np.asarray(logistic_q["theta"], dtype=float)

    return theta_q, q_mean, q_std, rbf_svm


st.set_page_config(
    page_title="Transmission Line Fault Detection",
    page_icon="⚡",
    layout="centered",
)

st.title("⚡ Three-Phase Transmission Line Fault Detection")
st.write(
    "Enter the six electrical measurements to compare "
    "the Set Q Logistic Regression and RBF SVM predictions."
)

st.info(
    "Enter measurements in the same units and format as the "
    "training dataset."
)

with st.form("fault_detection_form"):
    st.subheader("Electrical measurements")

    col1, col2 = st.columns(2)

    with col1:
        ia = st.number_input("Ia", value=0.0, format="%.6f")
        ib = st.number_input("Ib", value=0.0, format="%.6f")
        ic = st.number_input("Ic", value=0.0, format="%.6f")

    with col2:
        va = st.number_input("Va", value=0.0, format="%.6f")
        vb = st.number_input("Vb", value=0.0, format="%.6f")
        vc = st.number_input("Vc", value=0.0, format="%.6f")

    submitted = st.form_submit_button(
        "Detect Fault",
        type="primary",
        use_container_width=True,
    )


if submitted:
    try:
        theta_q, q_mean, q_std, rbf_svm = load_models()

        raw = np.array(
            [[ia, ib, ic, va, vb, vc]],
            dtype=float,
        )

        # Set Q: six raw features plus their squares
        q_features = np.column_stack([raw, raw ** 2])
        q_scaled = (q_features - q_mean) / q_std

        q_with_intercept = np.column_stack(
            [np.ones(len(q_scaled)), q_scaled]
        )

        logistic_prediction = int(
            predict(q_with_intercept, theta_q)[0]
        )

        # RBF SVM was trained on standardised Set R features
        # Use the training-set mean/std used by prepare.py
        train_df = pd.read_csv(DATA_DIR / "train_raw.csv")
        train_features = train_df[FEATURES].to_numpy(dtype=float)

        means_r = train_features.mean(axis=0)
        stds_r = train_features.std(axis=0, ddof=1)
        r_scaled = (raw - means_r) / stds_r

        svm_prediction = int(rbf_svm.predict(r_scaled)[0])

        st.subheader("Prediction results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.markdown("**Set Q Logistic Regression**")
            if logistic_prediction == 1:
                st.error("Fault detected")
            else:
                st.success("No fault detected")

        with result_col2:
            st.markdown("**RBF SVM**")
            if svm_prediction == 1:
                st.error("Fault detected")
            else:
                st.success("No fault detected")

        if logistic_prediction != svm_prediction:
            st.warning(
                "The models disagree. Review the measurements "
                "and consult the appropriate engineering process."
            )

        st.caption(
            "These are model predictions for educational use, "
            "not a substitute for validated protection equipment."
        )

    except FileNotFoundError as error:
        st.error(
            f"Required model or data file was not found: {error}"
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
