import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

st.title("⚙️ Predictive Maintenance Dashboard")
st.write("Input current machine telemetry parameters to predict potential failure.")

# Sidebar user inputs for live telemetry data
st.sidebar.header("Live Telemetry Input")
air_temp = st.sidebar.slider("Air Temperature (K)", 295.0, 305.0, 300.0)
process_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 310.0)
rot_speed = st.sidebar.slider("Rotational Speed (rpm)", 1100, 2800, 1500)
torque = st.sidebar.slider("Torque (Nm)", 3.0, 76.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 250, 100)

# Dropdowns for Machine Quality Type
machine_type = st.sidebar.selectbox("Machine Quality Type", ["Low (L)", "Medium (M)", "High (H)"])
type_m = 1 if machine_type == "Medium (M)" else 0
type_h = 1 if machine_type == "High (H)" else 0

# Calculated Features matching our notebook's engineering logic
temp_diff = process_temp - air_temp
power = torque * rot_speed

# Construct a single row DataFrame matching the exact training columns
input_data = pd.DataFrame([{
    'Air_Temp': air_temp,
    'Process_Temp': process_temp,
    'Rotational_Speed': rot_speed,
    'Torque': torque,
    'Tool_Wear': tool_wear,
    'Type_M': type_m,
    'Type_H': type_h,
    'Temp_Diff': temp_diff,
    'Power': power
}])

# Layout Columns
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Calculated Temp Difference", value=f"{temp_diff:.2f} K")
with col2:
    st.metric(label="Calculated Power Metric", value=f"{power:,} units")

# Trigger prediction on button click
# Note: In a production app, you would load a pre-trained model file (.json / .pkl)
if st.button("Analyze Equipment Health"):
    # Quick mock prediction logic for illustration since the full model state isn't saved here
    # An ideal practice is using model.save_model() in the notebook and loading it here!
    risk_factor = (tool_wear / 250.0) * 0.4 + (temp_diff / 20.0) * 0.4 + (torque / 76.0) * 0.2

    st.subheader("Analysis Verdict")
    if risk_factor > 0.65:
        st.error(f"🚨 ALERT: High risk of failure detected! Immediate maintenance recommended. (Risk Index: {risk_factor:.2f})")
    elif risk_factor > 0.4:
        st.warning(f"⚠️ WARNING: Moderate wear detected. Schedule inspection soon. (Risk Index: {risk_factor:.2f})")
    else:
        st.success(f"✅ NORMAL: Equipment operating within safe parameters. (Risk Index: {risk_factor:.2f})")
