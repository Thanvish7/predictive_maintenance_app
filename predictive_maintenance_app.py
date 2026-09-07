
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "predictive_maintenance_model.json"
METADATA_PATH = BASE_DIR / "model_metadata.json"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. "
        "Run the predictive_maintenance.ipynb notebook first "
        "to generate predictive_maintenance_model.json."
    )
    st.stop()


if not METADATA_PATH.exists():
    st.error(
        "Model metadata not found. "
        "Run the predictive_maintenance.ipynb notebook first."
    )
    st.stop()


model = load_model()
metadata = load_metadata()


# ============================================================
# TITLE
# ============================================================

st.title("Predictive Maintenance Dashboard")

st.write(
    "Predict the probability of machine failure using "
    "a trained XGBoost classification model."
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Precision",
        f"{metadata['precision']:.2%}"
    )

with col2:
    st.metric(
        "Recall",
        f"{metadata['recall']:.2%}"
    )

with col3:
    st.metric(
        "F1 Score",
        f"{metadata['f1_score']:.2%}"
    )

with col4:
    st.metric(
        "ROC-AUC",
        f"{metadata['roc_auc']:.4f}"
    )

st.divider()


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Machine Telemetry")

air_temp = st.sidebar.slider(
    "Air Temperature (K)",
    min_value=295.0,
    max_value=305.0,
    value=300.0,
    step=0.1
)

process_temp = st.sidebar.slider(
    "Process Temperature (K)",
    min_value=305.0,
    max_value=315.0,
    value=310.0,
    step=0.1
)

rot_speed = st.sidebar.slider(
    "Rotational Speed (rpm)",
    min_value=1100,
    max_value=2800,
    value=1500,
    step=10
)

torque = st.sidebar.slider(
    "Torque (Nm)",
    min_value=3.0,
    max_value=76.0,
    value=40.0,
    step=0.1
)

tool_wear = st.sidebar.slider(
    "Tool Wear (min)",
    min_value=0,
    max_value=250,
    value=100,
    step=1
)

machine_type = st.sidebar.selectbox(
    "Machine Quality Type",
    ["Low (L)", "Medium (M)", "High (H)"]
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

temp_diff = process_temp - air_temp
power = torque * rot_speed

type_l = 1 if machine_type == "Low (L)" else 0
type_m = 1 if machine_type == "Medium (M)" else 0


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame([{
    "Air_Temp": air_temp,
    "Process_Temp": process_temp,
    "Rotational_Speed": rot_speed,
    "Torque": torque,
    "Tool_Wear": tool_wear,
    "Type_L": type_l,
    "Type_M": type_m,
    "Temp_Diff": temp_diff,
    "Power": power
}])


# Make absolutely sure feature order matches training
feature_order = [
    "Air_Temp",
    "Process_Temp",
    "Rotational_Speed",
    "Torque",
    "Tool_Wear",
    "Type_L",
    "Type_M",
    "Temp_Diff",
    "Power"
]

input_data = input_data.reindex(
    columns=feature_order,
    fill_value=0
)


# ============================================================
# DISPLAY CURRENT TELEMETRY
# ============================================================

st.subheader("Current Machine Telemetry")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Air Temperature",
        f"{air_temp:.1f} K"
    )

with col2:
    st.metric(
        "Process Temperature",
        f"{process_temp:.1f} K"
    )

with col3:
    st.metric(
        "Rotational Speed",
        f"{rot_speed:,} rpm"
    )

with col4:
    st.metric(
        "Tool Wear",
        f"{tool_wear} min"
    )


col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Temperature Difference",
        f"{temp_diff:.2f} K"
    )

with col2:
    st.metric(
        "Power",
        f"{power:,.0f}"
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Analyze Equipment Health",
    type="primary",
    use_container_width=True
):

    # Actual XGBoost prediction
    failure_probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = model.predict(
        input_data
    )[0]


    # ========================================================
    # DISPLAY FAILURE PROBABILITY
    # ========================================================

    st.subheader("Failure Risk")

    probability_percent = failure_probability * 100

    st.progress(
        float(failure_probability),
        text=f"Failure Probability: {probability_percent:.2f}%"
    )


    # ========================================================
    # HEALTH STATUS
    # ========================================================

    if failure_probability >= 0.65:

        status = "CRITICAL"
        recommendation = (
            "High probability of machine failure. "
            "Immediate inspection and maintenance is recommended."
        )

        st.error(
            f"CRITICAL ALERT\n\n"
            f"Predicted failure probability: "
            f"{probability_percent:.2f}%"
        )

    elif failure_probability >= 0.25:

        status = "WARNING"
        recommendation = (
            "Moderate failure probability detected. "
            "Schedule an inspection and monitor machine telemetry."
        )

        st.warning(
            f"WARNING\n\n"
            f"Predicted failure probability: "
            f"{probability_percent:.2f}%"
        )

    else:

        status = "NORMAL"
        recommendation = (
            "Machine is operating within the model's "
            "low-risk range. Continue normal monitoring."
        )

        st.success(
            f"NORMAL\n\n"
            f"Predicted failure probability: "
            f"{probability_percent:.2f}%"
        )


    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader("Maintenance Recommendation")

    st.info(recommendation)


    # ========================================================
    # PREDICTION DETAILS
    # ========================================================

    st.subheader("Prediction Details")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Model Prediction",
            "FAILURE" if prediction == 1 else "NORMAL"
        )

    with result_col2:
        st.metric(
            "Failure Probability",
            f"{probability_percent:.2f}%"
        )

    with result_col3:
        st.metric(
            "Health Status",
            status
        )


    # ========================================================
    # INPUT DATA
    # ========================================================

    with st.expander("View Model Input"):

        display_data = input_data.copy()

        st.dataframe(
            display_data,
            use_container_width=True
        )
