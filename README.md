# End-to-End Predictive Maintenance and Telemetry Dashboard

An interactive machine learning application built to predict industrial equipment failures before they happen. This project utilizes sensor telemetry data to classify machinery operational status, generating live risk index reports to minimize factory downtime and repair costs.

Live App Link: https://thanvishpm.streamlit.app/

---

## Project Overview

In industrial manufacturing, unexpected equipment breakdowns lead to massive losses in production capacity and costly reactive repairs. This project addresses this challenge by analyzing historical sensor metrics to detect anomalies and degrading machine performance.

The project features a full end-to-end data pipeline, from custom feature engineering to class-imbalance-aware modeling, culminating in a live-updating user dashboard for factory technicians and management.

---

## Dataset and Features

The model is trained on the industry-benchmark AI4I 2020 Predictive Maintenance Dataset (sourced from the UCI Machine Learning Repository), which reflects 10,000 operational data points across real-world physical boundaries.

Key telemetry inputs utilized:
* Air Temperature (K) and Process Temperature (K)
* Rotational Speed (rpm) and Torque (Nm)
* Tool Wear (min) - accumulated operation duration
* Machine Quality Type - variants segmented into Low (L), Medium (M), and High (H)

### Advanced Feature Engineering
To improve predictive capability, custom physical interaction features were introduced:
1. Temperature Differential (Temp_Diff): Measures cooling efficiency over active run cycles.
2. Power Metric Approximations (Power): Captures multi-variable strain patterns using structural engineering approximations (Torque multiplied by Rotational Speed).

---

## Machine Learning Pipeline

* Handling Class Imbalance: Industrial asset datasets are naturally imbalanced (failures occur less than 4 percent of the time). This model utilizes an XGBoost Classifier configured with dynamic scale positive weighting (scale_pos_weight) to penalize missed critical failures.
* Temporal Evaluation Validation: A chronological train-test split (80/20 sequential window) was applied to prevent future-data leakage, simulating realistic deployment contexts.
* Key Evaluation Focus: Prioritized Recall and ROC-AUC metrics over basic accuracy to ensure maximum coverage of critical faults without generating excessive false alarms.

---

## App Interface Architecture

The final interactive web app is built with Streamlit. It takes inputs across operational limits and converts sensor configurations into definitive maintenance recommendations:
* NORMAL: Equipment operating reliably within standard physical boundaries.
* WARNING: Elevated asset stress or tool wear detected; inspection scheduling suggested.
* CRITICAL ALERT: High-probability breakdown profile identified; immediate shutdown or repair required.

---

## Local Installation and Setup

Follow these quick instructions to run the application locally on your computer:

1. Clone this repository:
   git clone https://github.com
   cd predictive-maintenance-app

2. Install the necessary system dependencies:
   python -m pip install -r requirements.txt

3. Launch your interactive Streamlit dashboard:
   python -m streamlit run app.py

---

## Technologies Used

* Language: Python
* Development environment: Jupyter Notebook
* Data and ML Pipelines: Pandas, NumPy, Scikit-Learn, XGBoost
* Visualization: Matplotlib, Seaborn
* Deployment Architecture: Streamlit Cloud
